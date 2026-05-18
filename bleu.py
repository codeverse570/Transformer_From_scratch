"""
bleu_eval.py  –  BLEU evaluation with beam search for your transformer.

Drop this file next to your main training script and run:
    python bleu_eval.py

Flags you can tweak at the bottom:
    BEAM_SIZE   – beam width (4-6 is the sweet spot, 1 = greedy)
    LENGTH_PEN  – length penalty exponent (0.6 is standard)
    MAX_EVAL    – how many val samples to score (None = all)
"""

import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from tokenizers import Tokenizer

from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Mask helpers (same as main.py)
# ─────────────────────────────────────────────────────────────────────────────

def _mask_k(x: torch.Tensor) -> torch.Tensor:
    return (x == 0).unsqueeze(1).unsqueeze(2)   # (B,1,1,T)

def _mask_q(x: torch.Tensor) -> torch.Tensor:
    return (x == 0).unsqueeze(1).unsqueeze(3)   # (B,1,T,1)


# ─────────────────────────────────────────────────────────────────────────────
# Greedy predict (kept for quick comparison / debugging)
# ─────────────────────────────────────────────────────────────────────────────

def predict_greedy(enc_tensor, encoder, decoder, max_len=128):
    """Returns generated token-ID list (BOS/EOS/PAD stripped)."""
    encoder.val(); decoder.val()

    enc_k = _mask_k(enc_tensor)
    enc_q = _mask_q(enc_tensor)
    E = encoder.fit_pre(enc_tensor, enc_k | enc_q)

    start = torch.zeros((1, max_len), dtype=torch.long, device=device)
    start[0, 0] = 2  # BOS

    generated = []
    for i in range(max_len - 1):
        dec_k = _mask_k(start)
        dec_q = _mask_q(start)
        logits = decoder.fit_pre(start, E, dec_k | dec_q, enc_k | dec_q)
        next_token = logits[0, i].argmax().item()

        if next_token in (3, 0):   # EOS or PAD
            break
        generated.append(next_token)
        start[0, i + 1] = next_token

    encoder.clear_memory()
    decoder.clear_memory()
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Beam search
# ─────────────────────────────────────────────────────────────────────────────

def beam_search(enc_tensor, encoder, decoder,
                beam_size=5, max_len=128, length_penalty=0.6):
    """
    Beam search decoder.

    Args:
        enc_tensor    : (1, T_e) int tensor already on device
        encoder       : your Encoder instance
        decoder       : your Decoder instance
        beam_size     : number of beams (4-6 recommended)
        max_len       : maximum output length
        length_penalty: exponent α in  score / len^α  (0.6 = Google standard)

    Returns:
        list[int]  – best hypothesis token IDs, BOS/EOS stripped
    """
    encoder.val(); decoder.val()

    # ── encode source once ────────────────────────────────────────────────
    enc_k = _mask_k(enc_tensor)                          # (1,1,1,T_e)
    enc_q = _mask_q(enc_tensor)                          # (1,1,T_e,1)
    E = encoder.fit_pre(enc_tensor, enc_k | enc_q)       # (1,T_e,d_model)

    # ── expand encoder output to beam_size copies ─────────────────────────
    # Shape: (beam_size, T_e, d_model) — avoids re-running encoder per step
    E_exp   = E.expand(beam_size, -1, -1)               # view, no copy
    enc_k_exp = enc_k.expand(beam_size, -1, -1, -1)     # (B,1,1,T_e)

    # ── beam state ────────────────────────────────────────────────────────
    # Each beam: [token_ids]   (including BOS)
    # Each score: cumulative log-prob (higher = better)
    beams      = [[2]]          # [ [BOS] ]
    scores     = [0.0]          # one score per beam
    completed  = []             # finished beams  [(score, [ids])]

    for step in range(max_len - 1):
        n_active = len(beams)

        # ── build batch decoder input ─────────────────────────────────────
        # Pad every live beam to max_len; only n_active rows needed
        inp = torch.zeros((n_active, max_len), dtype=torch.long, device=device)
        for b, seq in enumerate(beams):
            inp[b, :len(seq)] = torch.tensor(seq, device=device)

        dec_k = _mask_k(inp)                             # (n_active,1,1,T_d)
        dec_q = _mask_q(inp)                             # (n_active,1,T_d,1)

        # Slice encoder expansions to n_active (handles last step if beams
        # were pruned below beam_size when some completed early)
        E_b     = E_exp    [:n_active]
        enc_k_b = enc_k_exp[:n_active]

        # cross-attention mask: (n_active, 1, T_d, T_e)
        cross_mask = enc_k_b | dec_q

        logits = decoder.fit_pre(inp, E_b, dec_k | dec_q, cross_mask)
        # logits: (n_active, T_d, vocab)

        # ── only look at position = current output position ───────────────
        pos       = step          # position of the token we just placed
        step_logits = logits[:, pos, :]          # (n_active, vocab)
        log_probs   = F.log_softmax(step_logits, dim=-1)  # (n_active, vocab)

        # free decoder activations immediately (we only needed logits)
        decoder.clear_memory()

        # ── expand each beam by top-k tokens ─────────────────────────────
        all_candidates = []   # (score, beam_idx, token)

        topk_lp, topk_tok = torch.topk(log_probs, beam_size, dim=-1)
        # topk_lp / topk_tok : (n_active, beam_size)

        for b in range(n_active):
            if beams[b][-1] == 3:
                # this beam already ended — carry it forward unchanged
                all_candidates.append((scores[b], b, 3))
                continue
            for k in range(beam_size):
                token = topk_tok[b, k].item()
                lp    = topk_lp [b, k].item()
                all_candidates.append((scores[b] + lp, b, token))

        # ── select top beam_size candidates ──────────────────────────────
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        top = all_candidates[:beam_size]

        new_beams  = []
        new_scores = []
        for cand_score, b_idx, token in top:
            new_seq = beams[b_idx] + [token]
            if token == 3 or len(new_seq) >= max_len:
                completed.append((cand_score, new_seq))
            else:
                new_beams.append(new_seq)
                new_scores.append(cand_score)

        beams  = new_beams
        scores = new_scores

        # stop early if all beams finished
        if not beams:
            break

        # stop once we have enough completed beams
        if len(completed) >= beam_size:
            break

    # flush remaining live beams into completed
    for seq, sc in zip(beams, scores):
        completed.append((sc, seq))

    encoder.clear_memory()

    if not completed:
        return []

    # ── length-penalised reranking ────────────────────────────────────────
    def _penalised(score, seq):
        # exclude BOS from length count
        length = max(len(seq) - 1, 1)
        return score / (length ** length_penalty)

    best_score, best_seq = max(completed, key=lambda x: _penalised(x[0], x[1]))

    # strip BOS (index 0) and EOS (token 3)
    return [t for t in best_seq[1:] if t != 3]


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python BLEU  (no sacrebleu required)
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(hypotheses, references, max_n=4):
    """
    Corpus-level BLEU-{1..max_n}.

    hypotheses : list of token-ID lists  (model output, special tokens stripped)
    references : list of token-ID lists  (gold, special tokens stripped)
    Returns dict with keys BP, BLEU-1..4, P1..4
    """
    clip_counts  = Counter()
    total_counts = Counter()
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ng  = Counter(_ngrams(hyp, n))
            ref_ng  = Counter(_ngrams(ref, n))
            clipped = {ng: min(c, ref_ng[ng]) for ng, c in hyp_ng.items()}
            clip_counts [n] += sum(clipped.values())
            total_counts[n] += sum(hyp_ng.values())

    scores = {}
    for n in range(1, max_n + 1):
        scores[n] = clip_counts[n] / total_counts[n] if total_counts[n] else 0.0

    bp = (1.0 if hyp_len >= ref_len
          else 0.0 if hyp_len == 0
          else math.exp(1 - ref_len / hyp_len))

    def _bleu_n(up_to_n):
        logs = [math.log(scores[k]) if scores[k] > 0 else float("-inf")
                for k in range(1, up_to_n + 1)]
        if any(l == float("-inf") for l in logs):
            return 0.0
        return bp * math.exp(sum(logs) / up_to_n)

    return {
        "BP":     bp,
        "BLEU-1": _bleu_n(1), "BLEU-2": _bleu_n(2),
        "BLEU-3": _bleu_n(3), "BLEU-4": _bleu_n(4),
        "P1": scores[1], "P2": scores[2],
        "P3": scores[3], "P4": scores[4],
    }


def strip_special(ids, bos=2, eos=3, pad=0):
    return [t for t in ids if t not in (bos, eos, pad)]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── config ────────────────────────────────────────────────────────────
    MODEL_PATH = "../workspace/transformer-8.pth"
    TOK_PATH   = "bpe_translation.json"
    VAL_ENC    = "translation_data/validation/validation_encoder_input.json"
    VAL_TGT    = "translation_data/validation/validation_decoder_target.json"

    BEAM_SIZE   = 5      # 1 = greedy, 4-6 = beam search sweet spot
    LENGTH_PEN  = 0.6    # Google's standard; higher = prefer longer outputs
    MAX_LEN     = 128
    MAX_EVAL    = 2000   # set None to run on full validation set

    # ── load ──────────────────────────────────────────────────────────────
    print("Loading model …")
    model     = torch.load(MODEL_PATH, weights_only=False)
    tokenizer = Tokenizer.from_file(TOK_PATH)

    def _load(path):
        with open(path, "rb") as f:
            return np.array(json.load(f))

    x_val_encoder = _load(VAL_ENC)
    x_val_target  = _load(VAL_TGT)

    if MAX_EVAL is not None:
        x_val_encoder = x_val_encoder[:MAX_EVAL]
        x_val_target  = x_val_target [:MAX_EVAL]

    # ── inference ─────────────────────────────────────────────────────────
    hypotheses_beam   = []
    hypotheses_greedy = []
    references        = []

    decode_fn = "beam" if BEAM_SIZE > 1 else "greedy"
    print(f"Running {decode_fn} search (beam={BEAM_SIZE}) "
          f"on {len(x_val_encoder)} samples …\n")

    for idx in range(len(x_val_encoder)):
        enc_ids = x_val_encoder[idx].tolist()
        enc_t   = torch.as_tensor([enc_ids], device=device)

        if BEAM_SIZE > 1:
            hyp_beam = beam_search(
                enc_t, model.encoder, model.decoder,
                beam_size=BEAM_SIZE, max_len=MAX_LEN,
                length_penalty=LENGTH_PEN,
            )
        else:
            hyp_beam = predict_greedy(enc_t, model.encoder, model.decoder, MAX_LEN)

        # also run greedy for a side-by-side delta
        hyp_gr  = predict_greedy(enc_t, model.encoder, model.decoder, MAX_LEN)
        ref_ids = strip_special(x_val_target[idx].tolist())

        hypotheses_beam.append(hyp_beam)
        hypotheses_greedy.append(hyp_gr)
        references.append(ref_ids)

        if (idx + 1) % 100 == 0:
            src_text  = tokenizer.decode([t for t in enc_ids if t != 0])
            beam_text = tokenizer.decode(hyp_beam)
            gr_text   = tokenizer.decode(hyp_gr)
            ref_text  = tokenizer.decode(ref_ids)
            print(f"[{idx+1}/{len(x_val_encoder)}]")
            print(f"  SRC  : {src_text}")
            print(f"  BEAM : {beam_text}")
            print(f"  GRDY : {gr_text}")
            print(f"  REF  : {ref_text}\n")

    # ── BLEU ──────────────────────────────────────────────────────────────
    s_beam  = bleu_score(hypotheses_beam,   references)
    s_grdy  = bleu_score(hypotheses_greedy, references)

    def _fmt(s):
        return (f"  BP={s['BP']:.4f}  "
                f"BLEU-1={s['BLEU-1']*100:.2f}  "
                f"BLEU-2={s['BLEU-2']*100:.2f}  "
                f"BLEU-3={s['BLEU-3']*100:.2f}  "
                f"BLEU-4={s['BLEU-4']*100:.2f}")

    print("=" * 60)
    print(f"Beam search  (beam={BEAM_SIZE}, lp={LENGTH_PEN})")
    print(_fmt(s_beam))
    print()
    print("Greedy (for comparison)")
    print(_fmt(s_grdy))
    print()
    delta = (s_beam['BLEU-4'] - s_grdy['BLEU-4']) * 100
    print(f"  Beam vs greedy BLEU-4 delta: {delta:+.2f}")
    print("=" * 60)

    # ── optional sacrebleu ────────────────────────────────────────────────
    try:
        from sacrebleu.metrics import BLEU as SacreBLEU
        metric   = SacreBLEU()
        hyp_text = [tokenizer.decode(h) for h in hypotheses_beam]
        ref_text = [tokenizer.decode(r) for r in references]
        sb = metric.corpus_score(hyp_text, [ref_text])
        print(f"\nsacreBLEU (beam): {sb}")
    except ImportError:
        print("\n(pip install sacrebleu for sacreBLEU score)")