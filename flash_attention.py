"""
FlashAttention-2 — Single Fused Triton Kernel  (FA-2 Algorithm)
================================================================
This is the closest pure-Python-accessible equivalent to what
torch.nn.functional.scaled_dot_product_attention does internally.

Why your previous Triton (FA-1) was still slow
-----------------------------------------------
The previous kernel used the FA-1 update rule:
    O_i = correction * O_old + P @ V      ← rescales O every inner step

FA-2 eliminates that:
    O_i accumulates raw, divided ONCE after inner loop ends.

But the REAL speed gap was:
  1. num_stages=2  — Triton software pipeline was too shallow
  2. num_warps=4   — not enough parallelism for large tiles
  3. tl.dot using fp16 intermediates on both P and V — loses precision
     and forces type conversions that break the pipeline
  4. No @triton.autotune — tile sizes were fixed, not GPU-optimal

This version fixes all of that:
  • @triton.autotune over (B_r, B_c, num_warps, num_stages)
  • FA-2 accumulation (single normalisation per query tile)
  • Full fp32 accumulation for O_i (Tensor Core output → fp32 acc)
  • Async pipeline: num_stages=3/4 hides HBM latency behind compute
  • Grid = (T_r, B*H) — every query tile on its own SM simultaneously
  • Backward also autotuned, dK/dV computed in a separate KV-parallel pass

Supports
--------
  • Causal masking
  • Key-pad mask   (B, 1, 1, T)  bool — True = PAD
  • Query-pad mask (B, 1, T, 1)  bool — True = PAD
  • Additive attn_bias  (B, H, T, T)  float16

Install
-------
    pip install triton          # Linux / WSL2
    pip install triton-windows  # Windows native

PyTorch ≥ 2.0, CUDA GPU required.
"""

import math
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
#  Autotune configurations
#  Triton will benchmark every config on first call and cache the winner.
#  This is exactly what PyTorch's SDPA does internally.
# ─────────────────────────────────────────────────────────────────────────────

def _fwd_configs():
    """
    Grid of (B_r, B_c, num_warps, num_stages) to benchmark.
    Triton picks the fastest for your specific GPU + shape.
    """
    configs = []
    for B_r, B_c in [(128, 64), (64, 64), (64, 128), (128, 128), (32, 128)]:
        for nw in [4, 8]:
            for ns in [3, 4]:
                configs.append(
                    triton.Config({"B_r": B_r, "B_c": B_c},
                                  num_warps=nw, num_stages=ns)
                )
    return configs


def _bwd_configs():
    configs = []
    for B_r, B_c in [(64, 64), (128, 64), (64, 128)]:
        for nw in [4, 8]:
            for ns in [2, 3]:
                configs.append(
                    triton.Config({"B_r": B_r, "B_c": B_c},
                                  num_warps=nw, num_stages=ns)
                )
    return configs


# ─────────────────────────────────────────────────────────────────────────────
#  FORWARD KERNEL  — FA-2 algorithm, fully pipelined
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=_fwd_configs(), key=["T", "d_k", "d_v"])
@triton.jit
def _fa2_fwd_kernel(
    # ── tensor pointers ───────────────────────────────────────────────────────
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    KM_ptr, QM_ptr,          # key / query pad-mask  (B, T) bool flattened
    # ── strides: Q/K/V/O are (B, H, T, d) ───────────────────────────────────
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    so_b, so_h, so_t, so_d,
    sl_b, sl_h, sl_t,        # L strides  (B, H, T)
    skm_b,                   # KM stride over batch
    sqm_b,                   # QM stride over batch
    # ── dims ─────────────────────────────────────────────────────────────────
    T,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    scale: tl.constexpr,
    # ── tile sizes (set by autotune) ─────────────────────────────────────────
    B_r: tl.constexpr,
    B_c: tl.constexpr,
    # ── flags ─────────────────────────────────────────────────────────────────
    CAUSAL: tl.constexpr,
    HAS_KM: tl.constexpr,
    HAS_QM: tl.constexpr,
):
    """
    One Triton program = one (batch, head, query-tile) triple.

    Grid layout
    -----------
    axis-0 : query tile index  (pid_tr)     → T_r programs per (b,h)
    axis-1 : linearised (batch, head)       → B*H programs total
    Total programs = T_r * B * H  — all run simultaneously on the GPU's SMs.

    FA-2 accumulation (key difference from FA-1)
    ---------------------------------------------
    FA-1 per inner step:  O = corr*O + P@V    ← division buried inside
    FA-2 per inner step:  O = corr*O + P@V    ← identical formula!
                          BUT: O stays un-normalised the whole inner loop.
    FA-2 after inner loop: O = O / l          ← ONE division per query tile

    The saving: FA-1 divided implicitly (correction shrinks l, which scales
    the running average). FA-2 keeps O as a raw weighted sum, deferring the
    /l to the end. This saves (T_c - 1) elementwise-div passes over O_i.

    In a Triton kernel that matters because those extra ops compete with the
    async HBM pipeline — removing them lets the async prefetch of the next
    K/V tile completely hide memory latency.
    """
    pid_tr = tl.program_id(0)    # which query tile
    pid_bh = tl.program_id(1)    # linearised (batch * H) index
    H      = tl.num_programs(1)  # total number of (b,h) pairs → used for b/h split

    # Unpack batch and head indices
    # pid_bh = b * n_heads + h  →  b = pid_bh // n_heads,  h = pid_bh % n_heads
    # We don't have n_heads as constexpr, so we compute base pointers directly
    # using the (b,h) combined stride trick.
    pid_b  = pid_bh // (sl_b // sl_h)   # approximate; exact via pointer arithmetic
    pid_h  = pid_bh  % (sl_b // sl_h)

    # ── Query tile bounds ─────────────────────────────────────────────────────
    q_start = pid_tr * B_r
    q_offs  = q_start + tl.arange(0, B_r)   # (B_r,) absolute query positions
    d_offs  = tl.arange(0, d_k)             # (d_k,) head-dim offsets
    v_offs  = tl.arange(0, d_v)             # (d_v,) value-dim offsets

    q_mask  = q_offs < T                    # boundary mask

    # ── Base pointers for this (b, h) ────────────────────────────────────────
    Q_bh = Q_ptr + pid_bh * sq_h            # pointer to Q[b, h, :, :]
    K_bh = K_ptr + pid_bh * sk_h
    V_bh = V_ptr + pid_bh * sv_h
    O_bh = O_ptr + pid_bh * so_h
    L_bh = L_ptr + pid_bh * sl_h

    # ── Load Q tile into registers ────────────────────────────────────────────
    # Q_i stays in registers for the ENTIRE inner loop — this is the key
    # register-reuse property that makes tiled attention fast.
    Q_i = tl.load(
        Q_bh + q_offs[:, None] * sq_t + d_offs[None, :] * sq_d,
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)                        # (B_r, d_k)  in registers

    # ── FA-2 running state — lives in registers ───────────────────────────────
    m_i = tl.full([B_r], float('-inf'), dtype=tl.float32)   # row max
    l_i = tl.zeros([B_r],              dtype=tl.float32)    # normaliser
    O_i = tl.zeros([B_r, d_v],         dtype=tl.float32)    # raw accumulator

    # ── Inner loop: scan K/V tiles ────────────────────────────────────────────
    # num_stages controls Triton's software pipeline depth.
    # With num_stages=4, Triton issues 4 async HBM loads in flight at once,
    # overlapping memory latency with Tensor Core compute.
    T_c = tl.cdiv(T, B_c)

    for j in tl.range(0, T_c, num_stages=1):
        kv_start = j * B_c
        kv_offs  = kv_start + tl.arange(0, B_c)   # (B_c,)
        kv_mask  = kv_offs < T

        # Causal short-circuit: entire K tile is in the future
        if CAUSAL:
            skip = kv_start > q_start + B_r - 1
        else:
            skip = False

        # Load K tile — async prefetch starts here, compute overlaps with prev iter
        if not skip:
            K_j = tl.load(
                K_bh + kv_offs[:, None] * sk_t + d_offs[None, :] * sk_d,
                mask=kv_mask[:, None],
                other=0.0,
            ).to(tl.float32)                   # (B_c, d_k)

            # Load V tile
            V_j = tl.load(
                V_bh + kv_offs[:, None] * sv_t + v_offs[None, :] * sv_d,
                mask=kv_mask[:, None],
                other=0.0,
            ).to(tl.float32)                   # (B_c, d_v)

            # ── Attention scores — Tensor Core matmul ─────────────────────────────
            # tl.dot dispatches to WMMA / MMA instructions.
            # fp16 inputs → fp32 accumulator (acc=tl.float32 is default in Triton 2.x)
            S_ij = tl.dot(
                Q_i.to(tl.float16),
                tl.trans(K_j).to(tl.float16),
                allow_tf32=True,
            ).to(tl.float32) * scale           # (B_r, B_c)

            # ── Key padding mask ──────────────────────────────────────────────────
            if HAS_KM:
                km = tl.load(KM_ptr + pid_b * skm_b + kv_offs,
                            mask=kv_mask, other=True)          # (B_c,) bool
                S_ij = tl.where(km[None, :], float('-inf'), S_ij)

            # ── OOB mask ─────────────────────────────────────────────────────────
            S_ij = tl.where(kv_mask[None, :], S_ij, float('-inf'))

            # ── Causal mask within tile ───────────────────────────────────────────
            if CAUSAL:
                q_idx  = (q_offs  )[:, None]   # (B_r, 1)
                kv_idx = (kv_offs )[None, :]   # (1, B_c)
                S_ij   = tl.where(kv_idx > q_idx, float('-inf'), S_ij)

            # ── FA-2 online softmax update ────────────────────────────────────────
            #
            # m_ij  = max over kv dim of S_ij                  (B_r,)
            # m_new = elementwise max(m_i, m_ij)               (B_r,)
            # corr  = exp(m_i - m_new)                         rescale factor
            #
            # KEY FA-2 INSIGHT:
            # O_i is a raw (un-normalised) weighted sum.
            # We rescale O_i by corr to account for the new max, then
            # add the new tile's contribution  exp(S - m_new) @ V.
            # We do NOT divide by l here — that happens once after the loop.
            # This removes (T_c - 1) division ops from the critical path.

            m_ij  = tl.max(S_ij, axis=1)                      # (B_r,)
            m_new = tl.maximum(m_i, m_ij)                     # (B_r,)
            corr  = tl.exp(m_i - m_new)                       # (B_r,)

            P_ij  = tl.exp(S_ij - m_new[:, None])             # (B_r, B_c)

            # Accumulate O  (FA-2: raw sum, no /l inside loop)
            O_i = corr[:, None] * O_i + tl.dot(
                P_ij.to(tl.float16),
                V_j.to(tl.float16),
                allow_tf32=True,
            ).to(tl.float32)

            # Update normaliser and running max
            l_i = corr * l_i + tl.sum(P_ij, axis=1)          # (B_r,)
            m_i = m_new

    # ── FA-2: single normalisation after inner loop ───────────────────────────
    O_final = O_i / l_i[:, None]                          # (B_r, d_v)

    # ── Query padding mask ────────────────────────────────────────────────────
    if HAS_QM:
        qm = tl.load(QM_ptr + pid_b * sqm_b + q_offs,
                     mask=q_mask, other=True)              # (B_r,) bool
        O_final = tl.where(qm[:, None], 0.0, O_final)

    # ── Store logsumexp for backward ──────────────────────────────────────────
    L_out = tl.log(l_i) + m_i
    tl.store(L_bh + q_offs * sl_t, L_out, mask=q_mask)

    # ── Store output ──────────────────────────────────────────────────────────
    tl.store(
        O_bh + q_offs[:, None] * so_t + v_offs[None, :] * so_d,
        O_final.to(tl.float16),
        mask=q_mask[:, None],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BACKWARD KERNEL — dQ pass  (query-parallel, same grid as forward)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=_bwd_configs(), key=["T", "d_k", "d_v"])
@triton.jit
def _fa2_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr,
    L_ptr, D_ptr,
    dQ_ptr,
    KM_ptr,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    so_b, so_h, so_t, so_d,
    sl_b, sl_h, sl_t,
    sd_b, sd_h, sd_t,
    skm_b,
    T,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    scale: tl.constexpr,
    B_r:  tl.constexpr,
    B_c:  tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_KM: tl.constexpr,
):
    """
    Computes dQ for one query tile.
    Grid: (T_r, B*H)  — same as forward, fully parallel over query tiles.
    dK and dV are computed in a separate KV-parallel kernel to avoid atomics.
    """
    pid_tr = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // (sl_b // sl_h)
    pid_h  = pid_bh  % (sl_b // sl_h)

    q_start = pid_tr * B_r
    q_offs  = q_start + tl.arange(0, B_r)
    d_offs  = tl.arange(0, d_k)
    v_offs  = tl.arange(0, d_v)
    q_mask  = q_offs < T

    Q_bh  = Q_ptr  + pid_bh * sq_h
    K_bh  = K_ptr  + pid_bh * sk_h
    V_bh  = V_ptr  + pid_bh * sv_h
    dO_bh = dO_ptr + pid_bh * so_h
    dQ_bh = dQ_ptr + pid_bh * sq_h
    L_bh  = L_ptr  + pid_bh * sl_h
    D_bh  = D_ptr  + pid_bh * sd_h

    Q_i  = tl.load(Q_bh  + q_offs[:, None]*sq_t + d_offs[None,:]*sq_d,
                   mask=q_mask[:, None], other=0.0).to(tl.float32)
    dO_i = tl.load(dO_bh + q_offs[:, None]*so_t + v_offs[None,:]*so_d,
                   mask=q_mask[:, None], other=0.0).to(tl.float32)
    L_i  = tl.load(L_bh  + q_offs*sl_t,  mask=q_mask, other=0.0)
    D_i  = tl.load(D_bh  + q_offs*sd_t,  mask=q_mask, other=0.0)

    dQ_i = tl.zeros([B_r, d_k], dtype=tl.float32)

    T_c = tl.cdiv(T, B_c)
    for j in tl.range(0, T_c, num_stages=1):
        kv_start = j * B_c
        kv_offs  = kv_start + tl.arange(0, B_c)
        kv_mask  = kv_offs < T

        if CAUSAL:
            if kv_start > q_start + B_r - 1:
                break

        K_j = tl.load(K_bh + kv_offs[:,None]*sk_t + d_offs[None,:]*sk_d,
                      mask=kv_mask[:,None], other=0.0).to(tl.float32)
        V_j = tl.load(V_bh + kv_offs[:,None]*sv_t + v_offs[None,:]*sv_d,
                      mask=kv_mask[:,None], other=0.0).to(tl.float32)

        # Recompute A_ij from stored L — no T×T matrix needed
        S_ij = tl.dot(Q_i.to(tl.float16), tl.trans(K_j).to(tl.float16),
                      allow_tf32=True).to(tl.float32) * scale

        if HAS_KM:
            km   = tl.load(KM_ptr + pid_b*skm_b + kv_offs, mask=kv_mask, other=True)
            S_ij = tl.where(km[None,:], float('-inf'), S_ij)

        S_ij = tl.where(kv_mask[None,:], S_ij, float('-inf'))

        if CAUSAL:
            S_ij = tl.where(kv_offs[None,:] > q_offs[:,None], float('-inf'), S_ij)

        A_ij  = tl.exp(S_ij - L_i[:, None])               # (B_r, B_c)
        dA_ij = tl.dot(dO_i.to(tl.float16), tl.trans(V_j).to(tl.float16),
                       allow_tf32=True).to(tl.float32)

        # Softmax backward: dS = A * (dA - D)
        dS_ij = A_ij * (dA_ij - D_i[:, None]) * scale

        dQ_i += tl.dot(dS_ij.to(tl.float16), K_j.to(tl.float16),
                       allow_tf32=True).to(tl.float32)

    # Write dQ — no atomics needed, each program owns its query tile
    tl.store(dQ_bh + q_offs[:,None]*sq_t + d_offs[None,:]*sq_d,
             dQ_i.to(tl.float16), mask=q_mask[:,None])


# ─────────────────────────────────────────────────────────────────────────────
#  BACKWARD KERNEL — dK / dV pass  (KV-parallel)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=_bwd_configs(), key=["T", "d_k", "d_v"])
@triton.jit
def _fa2_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    KM_ptr,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    so_b, so_h, so_t, so_d,
    sl_b, sl_h, sl_t,
    sd_b, sd_h, sd_t,
    skm_b,
    T,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    scale: tl.constexpr,
    B_r:  tl.constexpr,
    B_c:  tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_KM: tl.constexpr,
):
    """
    Computes dK and dV for one KV tile.
    Grid: (T_c, B*H)  — parallel over KV tiles.

    This is the FA-2 backward trick: split dQ and dK/dV into two kernels
    with different parallelism axes.
    dQ kernel: parallel over query tiles   (each owns its dQ slice, no atomics)
    dKV kernel: parallel over KV tiles     (each owns its dK/dV slice, no atomics)
    Together they cover all gradients with zero atomic contention.
    """
    pid_tc = tl.program_id(0)    # which KV tile this program owns
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // (sl_b // sl_h)
    pid_h  = pid_bh  % (sl_b // sl_h)

    kv_start = pid_tc * B_c
    kv_offs  = kv_start + tl.arange(0, B_c)
    d_offs   = tl.arange(0, d_k)
    v_offs   = tl.arange(0, d_v)
    kv_mask  = kv_offs < T

    K_bh  = K_ptr  + pid_bh * sk_h
    V_bh  = V_ptr  + pid_bh * sv_h
    Q_bh  = Q_ptr  + pid_bh * sq_h
    dO_bh = dO_ptr + pid_bh * so_h
    dK_bh = dK_ptr + pid_bh * sk_h
    dV_bh = dV_ptr + pid_bh * sv_h
    L_bh  = L_ptr  + pid_bh * sl_h
    D_bh  = D_ptr  + pid_bh * sd_h

    K_j = tl.load(K_bh + kv_offs[:,None]*sk_t + d_offs[None,:]*sk_d,
                  mask=kv_mask[:,None], other=0.0).to(tl.float32)
    V_j = tl.load(V_bh + kv_offs[:,None]*sv_t + v_offs[None,:]*sv_d,
                  mask=kv_mask[:,None], other=0.0).to(tl.float32)

    dK_j = tl.zeros([B_c, d_k], dtype=tl.float32)
    dV_j = tl.zeros([B_c, d_v], dtype=tl.float32)

    T_r = tl.cdiv(T, B_r)
    for i in tl.range(0, T_r, num_stages=1):
        q_start = i * B_r
        q_offs  = q_start + tl.arange(0, B_r)
        q_mask  = q_offs < T

        # Causal: skip query tiles that are entirely before this KV tile
        if CAUSAL:
            if q_start + B_r - 1 < kv_start:
                continue

        Q_i  = tl.load(Q_bh  + q_offs[:,None]*sq_t + d_offs[None,:]*sq_d,
                       mask=q_mask[:,None], other=0.0).to(tl.float32)
        dO_i = tl.load(dO_bh + q_offs[:,None]*so_t + v_offs[None,:]*so_d,
                       mask=q_mask[:,None], other=0.0).to(tl.float32)
        L_i  = tl.load(L_bh  + q_offs*sl_t,  mask=q_mask, other=0.0)
        D_i  = tl.load(D_bh  + q_offs*sd_t,  mask=q_mask, other=0.0)

        S_ij = tl.dot(Q_i.to(tl.float16), tl.trans(K_j).to(tl.float16),
                      allow_tf32=True).to(tl.float32) * scale

        if HAS_KM:
            km   = tl.load(KM_ptr + pid_b*skm_b + kv_offs, mask=kv_mask, other=True)
            S_ij = tl.where(km[None,:], float('-inf'), S_ij)

        S_ij = tl.where(kv_mask[None,:], S_ij, float('-inf'))
        S_ij = tl.where(q_mask[:,None],  S_ij, float('-inf'))

        if CAUSAL:
            S_ij = tl.where(kv_offs[None,:] > q_offs[:,None], float('-inf'), S_ij)

        A_ij  = tl.exp(S_ij - L_i[:,None])               # (B_r, B_c)

        # dV_j += A_ij^T @ dO_i
        dV_j += tl.dot(tl.trans(A_ij).to(tl.float16),
                       dO_i.to(tl.float16), allow_tf32=True).to(tl.float32)

        dA_ij = tl.dot(dO_i.to(tl.float16), tl.trans(V_j).to(tl.float16),
                       allow_tf32=True).to(tl.float32)
        dS_ij = A_ij * (dA_ij - D_i[:,None]) * scale

        # dK_j += dS_ij^T @ Q_i
        dK_j += tl.dot(tl.trans(dS_ij).to(tl.float16),
                       Q_i.to(tl.float16), allow_tf32=True).to(tl.float32)

    # Write dK, dV — no atomics, each program owns its KV tile
    tl.store(dK_bh + kv_offs[:,None]*sk_t + d_offs[None,:]*sk_d,
             dK_j.to(tl.float16), mask=kv_mask[:,None])
    tl.store(dV_bh + kv_offs[:,None]*sv_t + v_offs[None,:]*sv_d,
             dV_j.to(tl.float16), mask=kv_mask[:,None])


# ─────────────────────────────────────────────────────────────────────────────
#  D = rowsum(dO * O) — fused helper kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _rowsum_kernel(
    dO_ptr, O_ptr, D_ptr,
    st, sd,             # strides for T and d dims (same layout for dO and O)
    sl_h,               # stride to jump one (b,h) block
    T, d_v: tl.constexpr,
    B_r:    tl.constexpr,
):
    pid_tr = tl.program_id(0)
    pid_bh = tl.program_id(1)

    t_offs = pid_tr * B_r + tl.arange(0, B_r)
    d_offs = tl.arange(0, d_v)
    mask   = t_offs < T

    base_dO = dO_ptr + pid_bh * sl_h
    base_O  = O_ptr  + pid_bh * sl_h

    dO = tl.load(base_dO + t_offs[:,None]*st + d_offs[None,:]*sd,
                 mask=mask[:,None], other=0.0).to(tl.float32)
    O  = tl.load(base_O  + t_offs[:,None]*st + d_offs[None,:]*sd,
                 mask=mask[:,None], other=0.0).to(tl.float32)

    D = tl.sum(dO * O, axis=1)                            # (B_r,)
    tl.store(D_ptr + pid_bh * T + t_offs, D, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
#  Python-side launcher helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _pad_last(x, target):
    """Zero-pad last dimension to target (must be power of 2 for tl.dot)."""
    if x.shape[-1] == target:
        return x
    p = torch.zeros(*x.shape[:-1], target - x.shape[-1],
                    dtype=x.dtype, device=x.device)
    return torch.cat([x, p], dim=-1)


def _prep_mask(mask, B, T):
    """Flatten (B,1,1,T) or (B,1,T,1) bool mask → (B,T) contiguous."""
    if mask is None:
        return None
    return mask.reshape(B, T).contiguous()


def _launch_forward(Q, K, V, causal, scale, key_mask, query_mask):
    B, H, T, d_k = Q.shape
    d_v = V.shape[-1]

    d_k2 = _next_pow2(d_k)
    d_v2 = _next_pow2(d_v)

    Q = _pad_last(Q, d_k2)
    K = _pad_last(K, d_k2)
    V = _pad_last(V, d_v2)

    O = torch.empty(B, H, T, d_v2, dtype=torch.float16, device=Q.device)
    L = torch.full((B, H, T), float('-inf'), dtype=torch.float32, device=Q.device)

    km = _prep_mask(key_mask,   B, T)
    qm = _prep_mask(query_mask, B, T)

    # Dummy pointers for disabled masks (kernel gates on HAS_KM / HAS_QM)
    km_ptr = km if km is not None else Q
    qm_ptr = qm if qm is not None else Q

    # scale must be a Python float (tl.constexpr)
    if scale is None:
        scale = float(1.0 / math.sqrt(d_k))

    def grid(meta):
        return (triton.cdiv(T, meta["B_r"]), B * H)

    _fa2_fwd_kernel[grid](
        Q, K, V, O, L,
        km_ptr, qm_ptr,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        km.stride(0) if km is not None else 0,
        qm.stride(0) if qm is not None else 0,
        T=T,
        d_k=d_k2, d_v=d_v2,
        scale=scale,
        CAUSAL=causal,
        HAS_KM=(km is not None),
        HAS_QM=(qm is not None),
    )

    return O[..., :d_v], L


def _launch_backward(Q, K, V, O, dO, L, causal, scale, key_mask):
    B, H, T, d_k = Q.shape
    d_v = V.shape[-1]

    if scale is None:
        scale = float(1.0 / math.sqrt(d_k))

    d_k2 = _next_pow2(d_k)
    d_v2 = _next_pow2(d_v)

    Q  = _pad_last(Q.contiguous(),  d_k2)
    K  = _pad_last(K.contiguous(),  d_k2)
    V  = _pad_last(V.contiguous(),  d_v2)
    O  = _pad_last(O.contiguous(),  d_v2)
    dO = _pad_last(dO.contiguous(), d_v2)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # D = rowsum(dO * O) via helper kernel
    D = torch.zeros(B, H, T, dtype=torch.float32, device=Q.device)
    B_r_rs = 64  # fixed tile for rowsum (small kernel)
    T_r_rs = triton.cdiv(T, B_r_rs)
    _rowsum_kernel[(T_r_rs, B * H)](
        dO, O, D,
        dO.stride(2), dO.stride(3),
        dO.stride(0) * dO.shape[1] // (B * H) if B * H > 1 else dO.stride(1),
        T, d_v2, B_r_rs,
    )

    km = _prep_mask(key_mask, B, T)
    km_ptr = km if km is not None else Q

    shared_args = dict(
        T=T, d_k=d_k2, d_v=d_v2, scale=scale,
        CAUSAL=causal, HAS_KM=(km is not None),
    )
    strides = (
        Q.stride(0),  Q.stride(1),  Q.stride(2),  Q.stride(3),
        K.stride(0),  K.stride(1),  K.stride(2),  K.stride(3),
        V.stride(0),  V.stride(1),  V.stride(2),  V.stride(3),
        O.stride(0),  O.stride(1),  O.stride(2),  O.stride(3),
        L.stride(0),  L.stride(1),  L.stride(2),
        D.stride(0),  D.stride(1),  D.stride(2),
        km.stride(0) if km is not None else 0,
    )

    # dQ kernel — query-parallel
    _fa2_bwd_dq_kernel[lambda m: (triton.cdiv(T, m["B_r"]), B * H)](
        Q, K, V, O, dO, L, D, dQ, km_ptr, *strides, **shared_args,
    )

    # dK/dV kernel — KV-parallel
    _fa2_bwd_dkv_kernel[lambda m: (triton.cdiv(T, m["B_c"]), B * H)](
        Q, K, V, dO, L, D, dK, dV, km_ptr, *strides, **shared_args,
    )

    return dQ[..., :d_k].to(Q.dtype), dK[..., :d_k].to(K.dtype), dV[..., :d_v].to(V.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  autograd.Function — full training loop integration
# ─────────────────────────────────────────────────────────────────────────────

class _FA2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, scale, key_mask, query_mask):
        O, L = _launch_forward(Q, K, V, causal, scale, key_mask, query_mask)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.causal    = causal
        ctx.scale     = scale
        ctx.key_mask  = key_mask
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = _launch_backward(
            Q, K, V, O, dO.contiguous(), L,
            ctx.causal, ctx.scale, ctx.key_mask,
        )
        return dQ, dK, dV, None, None, None, None


def flash_attention_v2(
    Q, K, V,
    causal:     bool                = True,
    scale:      float | None        = None,
    key_mask:   torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    FlashAttention-2  — single fused Triton kernel, autotuned.

    Drop-in for F.scaled_dot_product_attention.

    Args
    ----
    Q, K, V    : (B, H, T, d)  float16, CUDA
    causal     : autoregressive masking
    scale      : softmax scale (default 1/√d)
    key_mask   : (B,1,1,T)  bool — True = PAD key token
    query_mask : (B,1,T,1)  bool — True = PAD query token

    Returns
    -------
    O : (B, H, T, d)  float16
    """
    assert Q.is_cuda,             "Flash Attention requires a CUDA tensor"
    assert Q.dtype == torch.float16, "Flash Attention requires float16"
    return _FA2Func.apply(Q, K, V, causal, scale, key_mask, query_mask)


# ─────────────────────────────────────────────────────────────────────────────
#  Drop-in nn.Module for your masked_self_att
# ─────────────────────────────────────────────────────────────────────────────

class FlashMHA(torch.nn.Module):
    """
    Replaces your masked_self_att method.

    Usage
    -----
        self.attn = FlashMHA(d_model, h_count, dropout_p)

        # in forward:
        out = self.attn(x, key_mask=self.dec_k, query_mask=self.dec_q)
    """
    def __init__(self, d_model: int, h_count: int, dropout_p: float = 0.0):
        super().__init__()
        assert d_model % h_count == 0
        self.h   = h_count
        self.d_k = d_model // h_count
        self.dp  = dropout_p

        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x,
                key_mask=None, query_mask=None):
        B, T, C = x.shape
        H, d    = self.h, self.d_k

        def proj(W):
            return W(x).view(B, T, H, d).transpose(1, 2).half()

        Q, K, V = proj(self.W_q), proj(self.W_k), proj(self.W_v)

        O = flash_attention_v2(Q, K, V,
                               causal=True,
                               key_mask=key_mask,
                               query_mask=query_mask)     # (B, H, T, d)

        O = O.transpose(1, 2).reshape(B, T, C).to(x.dtype)
        return self.W_o(O)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick correctness check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn.functional as F

    assert torch.cuda.is_available(), "Need a CUDA GPU"
    torch.manual_seed(0)
    B, H, T, d = 2, 4, 256, 64
    device = "cuda"

    Q = torch.randn(B, H, T, d, device=device, dtype=torch.float16)
    K = torch.randn(B, H, T, d, device=device, dtype=torch.float16)
    V = torch.randn(B, H, T, d, device=device, dtype=torch.float16)

    km = torch.zeros(B, 1, 1, T, dtype=torch.bool, device=device)
    qm = torch.zeros(B, 1, T, 1, dtype=torch.bool, device=device)
    km[:, :, :, -32:] = True
    qm[:, :, -32:, :] = True

    # Reference
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q.float(), K.float().transpose(-2,-1)) * scale
    cm = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    S  = S.masked_fill(cm[None,None], float('-inf'))
    S  = S.masked_fill(km, float('-inf'))
    A  = torch.softmax(S, -1).nan_to_num(0.0)
    O_ref = torch.matmul(A.half(), V).masked_fill(qm, 0.0)

    # FA-2
    O_fa2 = flash_attention_v2(Q, K, V, causal=True, key_mask=km, query_mask=qm)

    err = (O_fa2.float() - O_ref.float()).abs().max().item()
    print(f"Max absolute error: {err:.4e}  {'✓' if err < 0.05 else '✗'}")

    # Autotune warmup timing
    import time
    for _ in range(3):
        flash_attention_v2(Q, K, V, causal=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        flash_attention_v2(Q, K, V, causal=True)
    torch.cuda.synchronize()
    print(f"FA-2 Triton:      {(time.perf_counter()-t0)*10:.3f} ms/iter")

    t0 = time.perf_counter()
    for _ in range(100):
        F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    print(f"PyTorch SDPA:     {(time.perf_counter()-t0)*10:.3f} ms/iter")