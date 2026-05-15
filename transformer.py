import json
import math
import time

import numpy as np
from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import Decoder, calculate_validation_loss, predict
from adam import AdamCustom, SchedulerState
from encoder import Encoder
from typing import List, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Mask helpers
# Masks are computed ON THE FLY per-batch (bool, (B,1,1,T) or (B,1,T,1))
# so we never store N × 1 × 1 × T bool tensors for the whole dataset.
# ─────────────────────────────────────────────────────────────────────────────

def _mask_k(x: torch.Tensor) -> torch.Tensor:
    """Key-side pad mask  (B, 1, 1, T)  –  True where token == 0."""
    return (x == 0).unsqueeze(1).unsqueeze(2)


def _mask_q(x: torch.Tensor) -> torch.Tensor:
    """Query-side pad mask  (B, 1, T, 1)  –  True where token == 0."""
    return (x == 0).unsqueeze(1).unsqueeze(3)


def make_masks(enc: torch.Tensor, dec: torch.Tensor):
    """
    Returns three combined bool masks (no intermediate storage):
      enc_comb       (B, 1, T_e, T_e)  encoder self-attention
      dec_comb       (B, 1, T_d, T_d)  decoder self-attention
      dec_cross_comb (B, 1, T_d, T_e)  decoder cross-attention
    All computed in-place without storing the full-dataset masks.
    """
    enc_comb       = _mask_k(enc) | _mask_q(enc)          # (B,1,T_e,T_e)
    dec_comb       = _mask_k(dec) | _mask_q(dec)          # (B,1,T_d,T_d)
    dec_cross_comb = _mask_k(enc) | _mask_q(dec)          # (B,1,T_d,T_e)
    return enc_comb, dec_comb, dec_cross_comb


# ─────────────────────────────────────────────────────────────────────────────
# Transformer
# ─────────────────────────────────────────────────────────────────────────────

class Transformer:
    def __init__(self, voc_size, d_model, max_len, d_ff, h_count,
                 layers, batch_size, lr=0.0001, epochs=40):
        self.schedular  = SchedulerState(lr, 4000, 100000)
        self.d_model    = d_model

        self.emb        = nn.Embedding(voc_size, d_model, device=device)
        self.emb_ad     = AdamCustom(0.99, voc_size, d_model, 0.01,
                                     schedular=self.schedular)
        self.pos        = nn.Embedding(max_len, d_model, device=device)
        self.pos_ada    = AdamCustom(0.99, max_len, d_model, 0.01,
                                     scale=True, schedular=self.schedular)

        self.decoder = Decoder(d_model, d_ff, h_count, voc_size, max_len,
                               layers, batch_size, schedular=self.schedular)
        self.encoder = Encoder(d_model, h_count, d_ff, voc_size, max_len,
                               layers, batch_size, schedular=self.schedular)

        nn.init.normal_(self.emb.weight, mean=0,
                        std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.pos.weight, mean=0, std=0.01)

    # ── single training step ──────────────────────────────────────────────
    def fit(self, encoder_inputs, decoder_inputs, targets,
            enc_comb, dec_comb, dec_cross_comb):
        """
        enc_comb / dec_comb / dec_cross_comb are the three masks returned
        by make_masks() – computed lazily per batch in the training loop.
        """
        # share embeddings
        self.decoder.emb = self.emb
        self.decoder.pos = self.pos
        self.encoder.emb = self.emb
        self.encoder.pos = self.pos

        self.decoder.train()
        self.encoder.train()

        # ── forward ──────────────────────────────────────────────────────
        E    = self.encoder.fit_pre(encoder_inputs, enc_comb)
        prob = self.decoder.fit_pre(decoder_inputs, E,
                                    dec_comb, dec_cross_comb)
        # print(E.dtype)

        self.schedular.advance()

        # ── backward ─────────────────────────────────────────────────────
        del_E, loss, dec_emb_grad, dec_pos_grad, decoder_all_grads = \
            self.decoder.back_pre(targets, prob)

        # free the decoder logits tensor – no longer needed
        del prob
        
        enc_emb_grad, enc_pos_grad, encoder_all_grads = \
            self.encoder.back_pre(del_E, E, encoder_inputs)

        # free encoder output – no longer needed after back_pre
        del E, del_E

        # ── clip + update ─────────────────────────────────────────────────
        emb_grad = dec_emb_grad + enc_emb_grad
        pos_grad = dec_pos_grad + enc_pos_grad

        coef = self.clip_grad_norm_fast(
            decoder_all_grads + encoder_all_grads + [emb_grad, pos_grad]
        )

        # weight updates (encoders/decoders clear their own grad stores)
        self.encoder.update_weights(coef)
        self.decoder.update_weights(coef)

        with torch.no_grad():
            self.emb.weight -= self.emb_ad.grad(coef * emb_grad, self.emb)
            self.pos.weight -= self.pos_ada.grad(coef * pos_grad, self.emb)

        # explicitly free temporaries so CUDA can reuse the memory
        del emb_grad, pos_grad, decoder_all_grads, encoder_all_grads
        del dec_emb_grad, enc_emb_grad, dec_pos_grad, enc_pos_grad

        return loss

    # ── fused grad-norm clip ──────────────────────────────────────────────
    def clip_grad_norm_fast(
        self,
        grads: List[Optional[torch.Tensor]],
        max_norm: float = 1.0,
        eps: float = 1e-6,
    ) -> float:
        valid = [g for g in grads if g is not None and g.numel() > 0]
        if not valid:
            return 1.0
        # single fused kernel – replaces one-kernel-per-tensor loop
        norms      = torch._foreach_norm(valid, ord=2)
        total_norm = torch.stack(norms).pow(2).sum().sqrt().item()
        return min(1.0, max_norm / (total_norm + eps))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── load data ─────────────────────────────────────────────────────────
    def _load(path):
        with open(path, "rb") as f:
            return np.array(json.load(f))

    x_train_encoder   = _load("translation_data/train/train_encoder_input.json")
    x_train_decoder   = _load("translation_data/train/train_decoder_input.json")
    x_train_target    = _load("translation_data/train/train_decoder_target.json")
    x_val_encoder     = _load("translation_data/validation/validation_encoder_input.json")
    x_val_decoder     = _load("translation_data/validation/validation_decoder_input.json")
    x_val_target      = _load("translation_data/validation/validation_decoder_target.json")

    BATCH   = 64
    MAX_LEN = 128
    epoch   = 2

    model = Transformer(d_model=256, h_count=8, d_ff=512, voc_size=16000,
                        max_len=MAX_LEN, layers=3, batch_size=BATCH)
    # model = torch.load('./models/transformer-3.pth',weights_only=False)

    total_iteration = len(x_train_encoder) // BATCH
    tokenizer       = Tokenizer.from_file("bpe_translation.json")

    # move training data to GPU once; keep as int (bool masks built per-batch)
    x_train_encoder = torch.as_tensor(x_train_encoder).pin_memory()
    x_train_decoder = torch.as_tensor(x_train_decoder).pin_memory()
    x_train_target  = torch.as_tensor(x_train_target).long().pin_memory()
    x_val_encoder   = torch.as_tensor(x_val_encoder,   device=device)
    x_val_decoder   = torch.as_tensor(x_val_decoder,   device=device)
    x_val_target    = torch.as_tensor(x_val_target,    device=device).long()
    # for i in range(0,10):
    #     print(tokenizer.decode(x_val_encoder[i].tolist()))
    # ── NO pre-computed masks for the whole dataset ────────────────────────
    # Old code built  x_train_encoder_com_mask  etc. of shape (N,1,1,T)
    # and kept them all in VRAM.  For N=50k, T=128 that is
    #   50000 × 1 × 128 × 128 × 1 byte  =  819 MB  just for one mask.
    # We now call make_masks() inside the loop for each mini-batch instead.
    # predict(torch.as_tensor(x_train_encoder[1502:1503],device=device),torch.as_tensor(x_train_decoder[1502:1503],device=device),model.encoder,model.decoder)
    samples = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I would like to order a coffee.",
        "The quick brown fox jumps over the lazy dog.",
    ]


    start_time = time.perf_counter()
    while epoch != 61:
        total_loss = 0.0
        iteration  = 1

        for i in range(0, len(x_train_encoder) - BATCH, BATCH):
            enc_b = x_train_encoder[i : i + BATCH].to(device, non_blocking=True)
            dec_b = x_train_decoder[i : i + BATCH].to(device, non_blocking=True)
            tgt_b = x_train_target [i : i + BATCH].to(device, non_blocking=True)

            # ── masks computed HERE, for this batch only ──────────────────
            # Each is bool (64, 1, 1/T, T) – freed as soon as fit() returns

            with torch.no_grad():
             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                enc_comb, dec_comb, dec_cross_comb = make_masks(enc_b, dec_b)

                loss = model.fit(enc_b, dec_b, tgt_b,
                                 enc_comb, dec_comb, dec_cross_comb)

            # masks go out of scope here – CUDA memory released immediately
            del enc_comb, dec_comb, dec_cross_comb

            total_loss += loss
            iteration  += 1
        
            
            print(iteration)
            if iteration% 1000==0:
                elapsed = time.perf_counter() - start_time
                print(f"{iteration}-iter checkpoint: {elapsed:.1f}s")
                for sample in samples:
                    ids = tokenizer.encode(sample).ids
                    padded = np.array([ids + [0] * (MAX_LEN - len(ids))])
                    sample_t = torch.as_tensor(padded, device=device)
                    predict(sample_t,np.array([ids]),model.encoder,model.decoder)
                val_enc_k = _mask_k(x_val_encoder)
                val_enc_q = _mask_q(x_val_encoder)
                val_dec_k = _mask_k(x_val_decoder)
                val_dec_q = _mask_q(x_val_decoder)
                val_loss = calculate_validation_loss(
            model.encoder, model.decoder,
            x_val_encoder, x_val_decoder, x_val_target,
            val_enc_k, val_enc_q, val_dec_k, val_dec_q,
        )
                print(f"validation loss: {val_loss:.4f}")

        elapsed = time.perf_counter() - start_time
        print(f"epoch {epoch} done in {elapsed:.1f}s  "
              f"| avg loss: {total_loss / total_iteration:.4f}")

        
        

        epoch += 1

        # ── validation ────────────────────────────────────────────────────
        # Build val masks per-batch inside calculate_validation_loss too.
        # Pass raw tensors; let that function handle its own masks.
        val_enc_k = _mask_k(x_val_encoder)
        val_enc_q = _mask_q(x_val_encoder)
        val_dec_k = _mask_k(x_val_decoder)
        val_dec_q = _mask_q(x_val_decoder)

        val_loss = calculate_validation_loss(
            model.encoder, model.decoder,
            x_val_encoder, x_val_decoder, x_val_target,
            val_enc_k, val_enc_q, val_dec_k, val_dec_q,
        )
        print(f"validation loss: {val_loss:.4f}")

        # free val masks immediately
        del val_enc_k, val_enc_q, val_dec_k, val_dec_q

        for sample in samples:
            ids = tokenizer.encode(sample).ids
            padded = np.array([ids + [0] * (MAX_LEN - len(ids))])
            sample_t = torch.as_tensor(padded, device=device)
            predict(sample_t,np.array([ids]),model.encoder,model.decoder)
            # predict is a quick inference pass – no mask storage needed
        # torch.save(model, f"../workspace/models/transformer-{epoch}.pth")