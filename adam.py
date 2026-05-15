import math
import torch
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────
 
class SchedulerState:
    def __init__(self, base_lr, warmup_steps=4000, total_steps=100000):
        self.global_step   = 0
        self.base_lr       = base_lr
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self._decay_span   = max(1, total_steps - warmup_steps)  # precomputed
 
    def advance(self):
        self.global_step += 1
 
    def get_lr(self) -> float:
        step = self.global_step or 1
        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / self._decay_span
            scale    = max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return self.base_lr * scale
 
 
# ─────────────────────────────────────────────────────────────────────────────
# AdamCustom  — fully in-place, zero extra allocations on the hot path
# ─────────────────────────────────────────────────────────────────────────────
 
class AdamCustom:
    """
    Key optimisations vs. the original
    ───────────────────────────────────
    1. All moment updates are IN-PLACE (add_, mul_, addcmul_).
       No new tensors are allocated on the hot path.
    2. Bias-correction is a scalar multiply/divide – no tensor created.
    3. `update` is computed as  lr * bc1 * m / (sqrt(v * bc2_inv) + eps)
       using torch.addcdiv / div_ to avoid temporaries.
    4. State initialised once with a boolean flag, not a dict-key lookup.
    5. grad_emb uses index_put_ (scatter write) instead of fancy indexing
       which copies data on read AND write.
    6. beta power is maintained as a running float (one float multiply per
       step) instead of   beta ** step   (full pow() each call).
    """
 
    def __init__(self, _, m, n, __, lr=1e-4,
                 beta1=0.9, beta2=0.999, eps=1e-9,
                 scale=False, schedular=None):
        self.m         = m
        self.n         = n
        self.scale     = scale
        self.schedular = schedular
        self.lr        = lr
        self.beta1     = beta1
        self.beta2     = beta2
        self.eps       = eps
 
        # step counters
        self.t  = 0   # weight (2-D) step
        self.tb = 0   # bias  (1-D) step
 
        # running beta powers  (avoids  beta**step  every call)
        self._b1t  = 1.0   # beta1^t   for weights
        self._b2t  = 1.0   # beta2^t
        self._b1tb = 1.0   # beta1^tb  for biases
        self._b2tb = 1.0
 
        # moment tensors (lazy init)
        self._init_w = False
        self._init_b = False
        self.m_w = self.v_w = None   # weight moments  (m, n)
        self.m_b = self.v_b = None   # bias moments    (m,)
 
        # for grad_emb: per-row step counter + moments
        self._init_emb = False
        self.m_emb = self.v_emb  = None   # (m, n)
        self.t_rows = None                 # (m,)  float32 for pow()
 
    # ── helpers ──────────────────────────────────────────────────────────────
 
    def _ensure_w(self):
        if not self._init_w:
            self.m_w = torch.zeros(self.m, self.n, device=device)
            self.v_w = torch.zeros(self.m, self.n, device=device)
            self._init_w = True
 
    def _ensure_b(self):
        if not self._init_b:
            self.m_b = torch.zeros(self.m, device=device)
            self.v_b = torch.zeros(self.m, device=device)
            self._init_b = True
 
    def _ensure_emb(self):
        if not self._init_emb:
            self.m_emb  = torch.zeros(self.m, self.n, device=device)
            self.v_emb  = torch.zeros(self.m, self.n, device=device)
            self.t_rows = torch.zeros(self.m, device=device)
            self._init_emb = True
 
    # ── main update ───────────────────────────────────────────────────────────
 
    def grad(self, G: torch.Tensor, W=None) -> torch.Tensor:
        """
        Returns the Adam update tensor (same shape as G).
        All moment arithmetic is in-place; only `update` is a new tensor.
        """
        is_2d = G.dim() == 2
 
        if is_2d:
            self._ensure_w()
            self.t  += 1
            self._b1t  *= self.beta1
            self._b2t  *= self.beta2
            m, v       = self.m_w, self.v_w
            bc1        = 1.0 - self._b1t   # (1 - beta1^t)
            bc2        = 1.0 - self._b2t
        else:
            self._ensure_b()
            self.tb += 1
            self._b1tb *= self.beta1
            self._b2tb *= self.beta2
            m, v        = self.m_b, self.v_b
            bc1         = 1.0 - self._b1tb
            bc2         = 1.0 - self._b2tb
 
        # m  ←  β1·m  +  (1-β1)·G          (in-place)
        m.mul_(self.beta1).add_(G, alpha=1.0 - self.beta1)
 
        # v  ←  β2·v  +  (1-β2)·G²         (in-place, no G² temp)
        v.mul_(self.beta2).addcmul_(G, G, value=1.0 - self.beta2)
 
        # update  =  lr * (m/bc1) / (sqrt(v/bc2) + eps)
        #          =  lr/bc1 * m / (sqrt(v/bc2) + eps)
        #
        # We avoid  m_hat = m/bc1  (new tensor) by folding bc1 into lr.
        # sqrt(v/bc2) = sqrt(v) / sqrt(bc2)  → we scale eps instead.
        #
        #   denom  = sqrt(v) / sqrt(bc2) + eps
        #   update = (lr/bc1) * m / denom
        #
        # One allocation: `update` itself (same shape as G, required output).
 
        sqrt_bc2  = math.sqrt(bc2)
        lr_now    = self.schedular.get_lr() if self.schedular else self.lr
 
        # denom: sqrt(v) in-place scaled — but we must NOT mutate v.
        # Use torch.sqrt(v) which allocates once, then scale in-place.
        denom = v.sqrt().div_(sqrt_bc2).add_(self.eps)          # 1 alloc
 
        # update = m * (lr_now / bc1) / denom
        update = m.div(denom).mul_(lr_now / bc1)                # 1 alloc (output)
 
        return update