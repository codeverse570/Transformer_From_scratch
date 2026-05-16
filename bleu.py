"""
bleu_eval.py  –  BLEU evaluation for your transformer translation model.

Drop this file next to your main training script and run:
    python bleu_eval.py

It monkey-patches predict() to capture tokens instead of printing them,
then computes corpus-level BLEU-1 through BLEU-4 without any extra deps
(uses only the standard library + what you already have installed).
If sacrebleu is available it also prints sacreBLEU for comparison.
"""

import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from tokenizers import Tokenizer

# ── import your own modules (same as main.py) ──────────────────────────────
from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── re-use the same mask helpers from main.py ──────────────────────────────
def _mask_k(x):
    return (x == 0).unsqueeze(1).unsqueeze(2)

def _mask_q(x):
    return (x == 0).unsqueeze(1).unsqueeze(3)


# ─────────────────────────────────────────────────────────────────────────────
# A predict() that RETURNS tokens instead of printing them
# ─────────────────────────────────────────────────────────────────────────────

def predict_tokens(y, encoder, decoder, max_len=128):
    """
    Like your original predict() but:
      - returns the generated token-ID list (excluding BOS/EOS/PAD)
      - does not print anything
    """
    encoder.val()
    decoder.val()

    enc_pad_k = _mask_k(y)
    enc_pad_q = _mask_q(y)
    E = encoder.fit_pre(y, enc_pad_k | enc_pad_q)

    start = torch.zeros((1, max_len), dtype=torch.long, device=device)
    start[0, 0] = 2  # BOS

    generated = []
    for i in range(max_len - 1):
        pad_mask_k = _mask_k(start)
        pad_mask_q = _mask_q(start)
        logits = decoder.fit_pre(
            start, E,
            pad_mask_k | pad_mask_q,
            enc_pad_k   | pad_mask_q,
        )
        prob = F.softmax(logits, dim=-1)
        _, index = torch.max(prob, dim=-1)
        next_token = index[0, i].item()

        if next_token == 3:   # EOS
            break
        if next_token == 0:   # PAD – shouldn't happen, but guard anyway
            break

        generated.append(next_token)
        start[0, i + 1] = next_token

    encoder.clear_memory()
    decoder.clear_memory()
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python BLEU (no sacrebleu required)
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(hypotheses, references, max_n=4):
    """
    Corpus-level BLEU-{1..max_n}.

    hypotheses : list of token-ID lists  (your model output)
    references : list of token-ID lists  (gold targets, PAD/BOS/EOS stripped)
    Returns a dict  {1: float, 2: float, 3: float, 4: float}
    """
    clip_counts   = Counter()
    total_counts  = Counter()
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)

        for n in range(1, max_n + 1):
            hyp_ng  = Counter(_ngrams(hyp, n))
            ref_ng  = Counter(_ngrams(ref, n))
            clipped = {ng: min(c, ref_ng[ng]) for ng, c in hyp_ng.items()}
            clip_counts[n]  += sum(clipped.values())
            total_counts[n] += sum(hyp_ng.values())

    scores = {}
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            scores[n] = 0.0
        else:
            scores[n] = clip_counts[n] / total_counts[n]

    # brevity penalty
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    # BLEU-N = BP * exp( (1/N) * sum_{k=1}^{N} log Pk )
    # i.e. geometric mean of P1..PN, then multiply by BP.
    def _bleu_n(up_to_n):
        logs = [math.log(scores[k]) if scores[k] > 0 else float("-inf")
                for k in range(1, up_to_n + 1)]
        if any(l == float("-inf") for l in logs):
            return 0.0
        return bp * math.exp(sum(logs) / up_to_n)

    return {
        "BP":      bp,
        "BLEU-1":  _bleu_n(1),
        "BLEU-2":  _bleu_n(2),
        "BLEU-3":  _bleu_n(3),
        "BLEU-4":  _bleu_n(4),
        # raw precisions (unpenalised) – handy for debugging
        "P1": scores[1], "P2": scores[2],
        "P3": scores[3], "P4": scores[4],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strip special tokens from a padded target row
# ─────────────────────────────────────────────────────────────────────────────

def strip_special(ids, bos=2, eos=3, pad=0):
    """Remove BOS, EOS and PAD tokens; return plain list."""
    return [t for t in ids if t not in (bos, eos, pad)]


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    MAX_LEN   = 128
    # How many validation samples to evaluate.
    # Set to None to run on the full validation set (slow without a GPU!).
    MAX_EVAL  = 2000

    # ── load model ────────────────────────────────────────────────────────
    print("Loading model …")
    model = torch.load("./models/transformer-8.pth", weights_only=False)

    # ── load tokenizer ────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file("bpe_translation.json")

    # ── load validation data ──────────────────────────────────────────────
    def _load(path):
        with open(path, "rb") as f:
            return np.array(json.load(f))

    x_val_encoder = _load("translation_data/validation/validation_encoder_input.json")
    x_val_target  = _load("translation_data/validation/validation_decoder_target.json")

    if MAX_EVAL is not None:
        x_val_encoder = x_val_encoder[:MAX_EVAL]
        x_val_target  = x_val_target [:MAX_EVAL]

    # ── run inference ─────────────────────────────────────────────────────
    hypotheses, references = [], []

    print(f"Running inference on {len(x_val_encoder)} samples …\n")
    for idx in range(len(x_val_encoder)):
        enc_ids = x_val_encoder[idx].tolist()
        enc_t   = torch.as_tensor([enc_ids], device=device)

        hyp_ids = predict_tokens(enc_t, model.encoder, model.decoder, MAX_LEN)
        ref_ids = strip_special(x_val_target[idx].tolist())

        hypotheses.append(hyp_ids)
        references.append(ref_ids)

        # progress + spot-check every 50 samples
        if (idx + 1) % 50 == 0:
            src_text  = tokenizer.decode([t for t in enc_ids if t != 0])
            hyp_text  = tokenizer.decode(hyp_ids)
            ref_text  = tokenizer.decode(ref_ids)
            print(f"[{idx+1}/{len(x_val_encoder)}]")
            print(f"  SRC : {src_text}")
            print(f"  HYP : {hyp_text}")
            print(f"  REF : {ref_text}\n")

    # ── compute BLEU ──────────────────────────────────────────────────────
    scores = bleu_score(hypotheses, references)

    print("=" * 50)
    print("Corpus BLEU (custom implementation)")
    print("=" * 50)
    print(f"  Brevity Penalty : {scores['BP']:.4f}")
    print(f"  BLEU-1          : {scores['BLEU-1']*100:.2f}")
    print(f"  BLEU-2          : {scores['BLEU-2']*100:.2f}")
    print(f"  BLEU-3          : {scores['BLEU-3']*100:.2f}")
    print(f"  BLEU-4          : {scores['BLEU-4']*100:.2f}")
    print(f"\n  (raw precisions – before BP)")
    print(f"  P1={scores['P1']:.4f}  P2={scores['P2']:.4f}  "
          f"P3={scores['P3']:.4f}  P4={scores['P4']:.4f}")

    # ── optional: sacrebleu (more standard) ───────────────────────────────
    try:
        from sacrebleu.metrics import BLEU as SacreBLEU
        metric = SacreBLEU()
        hyp_texts = [tokenizer.decode(h) for h in hypotheses]
        ref_texts = [tokenizer.decode(r) for r in references]
        sb = metric.corpus_score(hyp_texts, [ref_texts])
        print("\n" + "=" * 50)
        print("sacreBLEU (if you trust this more)")
        print("=" * 50)
        print(f"  {sb}")
    except ImportError:
        print("\n(Install sacrebleu with  pip install sacrebleu  "
              "to also get a sacreBLEU score.)")