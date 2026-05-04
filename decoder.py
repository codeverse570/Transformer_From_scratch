import gc
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
# from AdamCustom import AdamCustom
from adam import AdamCustom
from utils.utils import count_tensors,debug_top_tensors
from tokenizers import Tokenizer
from dropout import Dropout
# from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = Tokenizer.from_file("bpe_translation.json")

from utils.utils import abs_mean
negative_inf = float('-inf')
torch.set_default_dtype(torch.float32)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, h_count, voc_size, sent_len, layers, batch_size,schedular):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.is_training= True
        self.d_k = d_model // h_count
        self.voc_size = voc_size
        self.h_count = h_count
        self.sent_len = sent_len
        self.layers = layers
        self.batch_size = batch_size
        self.dropout= {'emb':Dropout(0),'self_att_a':Dropout(),'self_att_out':Dropout(),'cross_att_a':Dropout(),'cross_att_out':Dropout(),'ff_layer_1':Dropout(),'ff_out':Dropout()}
        self.schedular=schedular
        self.D = []
        self.pad_mask=[]
        self.emb = nn.Embedding(voc_size, self.d_model,device=device)
        self.emb_ad = AdamCustom(0.99, self.voc_size, self.d_model, 0.01,schedular=self.schedular)
        self.pos = nn.Embedding(sent_len, self.d_model,device=device)
        self.pos_ada = AdamCustom(0.99, self.sent_len, self.d_model, 0.01,schedular=self.schedular)
        self.sin_pos= self.get_sinusoidal_positional_encoding(self.sent_len,self.d_model)
        self.W_q, self.W_k, self.W_v, self.W_o, \
        self.W_ff1, self.W_ff2, \
        self.W_norm_self_a, self.W_norm_self_b, \
        self.W_norm_cross_a, self.W_norm_cross_b, \
        self.W_norm_ff_a, self.W_norm_ff_b, \
        self.W_cross_q, self.W_cross_k, self.W_cross_v, self.W_cross_o = self.intialize_weights()

        self.W_q_ada, self.W_k_ada, self.W_v_ada, self.W_o_ada, \
        self.W_ff1_ada, self.W_ff2_ada, \
        self.W_norm_self_a_ada, self.W_norm_self_b_ada, \
        self.W_norm_cross_a_ada, self.W_norm_cross_b_ada, \
        self.W_norm_ff_a_ada, self.W_norm_ff_b_ada, \
        self.W_cross_q_ada, self.W_cross_k_ada, self.W_cross_v_ada, self.W_cross_o_ada = self.intialize_optimizers()

        # self.b_ff1 = [nn.Parameter(torch.zeros(self.d_ff,device=device)) for i in range(self.layers)]
        # self.b_ff2 = [nn.Parameter(torch.zeros(self.d_model,device=device)) for i in range(self.layers)]


        self.W_voc = nn.Linear(self.d_model, self.voc_size).to(device)
        self.W_voc_ada = AdamCustom(0.99, self.voc_size, self.d_model, 0.01,schedular=self.schedular)

        # ── Activation caches (cleared at start of every fit()) ──
        self.ff_output = []
        self.ff_inputs = []
        self.ff_layer_1_output = []
        self.cross_att_output = []
        self.cross_att_inputs = []
        self.cross_att_o = []
        self.cross_att_q = []
        self.cross_att_k = []
        self.cross_att_v = []
        self.cross_att_s = []
        self.cross_att_a = []
        self.cross_att_raw_a = []
        self.self_att_output = []
        self.self_att_inputs = []
        self.self_att_o = []
        self.self_att_q = []
        self.self_att_k = []
        self.self_att_v = []
        self.self_att_s = []
        self.self_att_a = []
        self.self_att_raw_a = []
        self.residual_self=[]

        # ── Per-batch state (set in fit, deleted in back) ──
        self.H = None
        self.prob = None
        self.E = None
        self.D = None

        # nn.init.xavier_uniform_(self.W_voc.weight)
        # nn.init.xavier_normal_(self.emb.weight)
        # nn.init.xavier_normal_(self.pos.weight)
        self.epsilon = 0.001
        self.causal_mask = torch.triu(torch.full((self.sent_len, self.sent_len), float('-inf'), device=device), diagonal=1)

    # ─────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────
    def intialize_optimizers(self):
        W_q_ada = []
        W_k_ada = []
        W_v_ada = []
        W_o_ada = []
        W_cross_q_ada = []
        W_cross_k_ada = []
        W_cross_v_ada = []
        W_cross_o_ada = []
        W_ff1_ada = []
        W_ff2_ada = []
        W_norm_self_a_ada = []
        W_norm_self_b_ada = []
        W_norm_cross_a_ada = []
        W_norm_cross_b_ada = []
        W_norm_ff_a_ada = []
        W_norm_ff_b_ada = []

        for i in range(self.layers):
            W_q_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_k_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_v_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_o_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_cross_q_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_cross_k_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_cross_v_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_cross_o_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_ff1_ada.append(AdamCustom(0.99, self.d_ff, self.d_model, 0.01,schedular=self.schedular))
            W_ff2_ada.append(AdamCustom(0.99, self.d_model, self.d_ff, 0.01,schedular=self.schedular))
            W_norm_self_a_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_norm_self_b_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_norm_cross_a_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_norm_cross_b_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_norm_ff_a_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))
            W_norm_ff_b_ada.append(AdamCustom(0.99, self.d_model, self.d_model, 0.01,schedular=self.schedular))

        return (W_q_ada, W_k_ada, W_v_ada, W_o_ada,
                W_ff1_ada, W_ff2_ada,
                W_norm_self_a_ada, W_norm_self_b_ada,
                W_norm_cross_a_ada, W_norm_cross_b_ada,
                W_norm_ff_a_ada, W_norm_ff_b_ada,
                W_cross_q_ada, W_cross_k_ada, W_cross_v_ada, W_cross_o_ada)
    
    def intialize_weights(self):
        W_q = []
        W_k = []
        W_v = []
        W_o = []
        W_cross_q = []
        W_cross_k = []
        W_cross_v = []
        W_cross_o = []
        W_ff1 = []
        W_ff2 = []
        W_norm_self_a = []
        W_norm_self_b = []
        W_norm_cross_a = []
        W_norm_cross_b = []
        W_norm_ff_a = []
        W_norm_ff_b = []

        for i in range(self.layers):


            layer_mat_W_q = nn.Linear(self.d_model, self.d_model).to(device)
            layer_mat_W_k = nn.Linear(self.d_model, self.d_model).to(device)
            layer_mat_W_v = nn.Linear(self.d_model, self.d_model).to(device)
            layer_mat_W_o = nn.Linear(self.d_model, self.d_model).to(device)

            layer_cross_W_q = nn.Linear(self.d_model, self.d_model).to(device)
            layer_cross_W_k = nn.Linear(self.d_model, self.d_model).to(device)
            layer_cross_W_v = nn.Linear(self.d_model, self.d_model).to(device)
            layer_cross_W_o = nn.Linear(self.d_model, self.d_model).to(device)

            layer_ff_W1 = nn.Linear(self.d_model, self.d_ff).to(device)
            layer_ff_W2 = nn.Linear(self.d_ff, self.d_model).to(device)

            norm_self_a = nn.Parameter(torch.ones(self.d_model, device=device))
            norm_self_b = nn.Parameter(torch.zeros(self.d_model, device=device))

            norm_cross_a = nn.Parameter(torch.ones(self.d_model, device=device))
            norm_cross_b = nn.Parameter(torch.zeros(self.d_model, device=device))

            norm_ff_a = nn.Parameter(torch.ones(self.d_model, device=device))
            norm_ff_b = nn.Parameter(torch.zeros(self.d_model, device=device))

            for m in [
                      layer_mat_W_k, layer_mat_W_q, layer_mat_W_v, layer_mat_W_o,
                      layer_cross_W_k, layer_cross_W_q, layer_cross_W_v, layer_cross_W_o,layer_ff_W1,layer_ff_W2]:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

            W_q.append(layer_mat_W_q)
            W_k.append(layer_mat_W_k)
            W_v.append(layer_mat_W_v)
            W_o.append(layer_mat_W_o)
            W_ff1.append(layer_ff_W1)
            W_ff2.append(layer_ff_W2)
            W_norm_self_a.append(norm_self_a)
            W_norm_self_b.append(norm_self_b)
            W_norm_cross_a.append(norm_cross_a)
            W_norm_cross_b.append(norm_cross_b)
            W_norm_ff_a.append(norm_ff_a)
            W_norm_ff_b.append(norm_ff_b)
            W_cross_q.append(layer_cross_W_q)
            W_cross_k.append(layer_cross_W_k)
            W_cross_o.append(layer_cross_W_o)
            W_cross_v.append(layer_cross_W_v)

        return (W_q, W_k, W_v, W_o,
                W_ff1, W_ff2,
                W_norm_self_a, W_norm_self_b,
                W_norm_cross_a, W_norm_cross_b,
                W_norm_ff_a, W_norm_ff_b,
                W_cross_q, W_cross_k, W_cross_v, W_cross_o)

    # ─────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────
 
    def fit_pre(self, X, E,E_pad_mask):
     with torch.no_grad():
        
        self.clear_memory()

        self.ff_output = []
        self.ff_inputs = []
        self.ff_layer_1_output = []

        self.cross_att_output = []
        self.cross_att_inputs = []
        self.cross_att_o = []
        self.cross_att_q = []
        self.cross_att_k = []
        self.cross_att_v = []
        self.cross_att_s = []
        self.cross_att_a = []
        self.cross_att_raw_a = []
        self.self_att_output = []
        self.self_att_inputs = []
        self.self_att_o = []
        self.self_att_q = []
        self.self_att_k = []
        self.self_att_v = []
        self.self_att_s = []
        self.self_att_a = []
        self.self_att_raw_a = []
        # print(self.emb.weight.shape)
        self.pad_mask_q= self.create_pad_mask_q(X)
        self.pad_mask_k = self.create_pad_mask_k(X)
        embeddings = self.emb.weight[X]* math.sqrt(self.d_model)
        self.D = X
        inputs = embeddings+ self.pos.weight[:len(X[0])]
        inputs= self.dropout['emb'].train(self.is_training).forward(inputs)
        # inputs = embeddings+ self.sin_pos[:len(X[0])]
        # inputs = embeddings
        self.E = E

        # NEW: store residual inputs (needed for correct gradients later)
        self.residual_self = []
        self.residual_cross = []
        self.residual_ff = []
        
        for layer in range(self.layers):

            # =========================
            # 🔹 SELF ATTENTION (Pre-LN)
            # =========================
            norm_inputs = self.layer_norm(
                inputs,
                self.W_norm_self_a[layer], self.W_norm_self_b[layer]
            )

            multi_self_att, multi_self_att_o, multi_self_att_Q, multi_self_att_K, \
            multi_self_att_V, multi_self_att_A, multi_self_att_raw_A = self.masked_self_att(norm_inputs, layer)

            self_out = inputs +self.dropout['self_att_out'].train(self.is_training).forward( multi_self_att)

            # =========================
            # 🔹 CROSS ATTENTION (Pre-LN)
            # =========================
            norm_self_out = self.layer_norm(
                self_out,
                self.W_norm_cross_a[layer], self.W_norm_cross_b[layer]
            )

            cross_self_att, cross_att_o, cross_att_Q, cross_att_K, cross_att_V, \
             cross_att_A,cross_att_raw_A= self.cross_attention(norm_self_out, E, layer,E_pad_mask)

            cross_out = self_out + self.dropout['cross_att_out'].train(self.is_training).forward(cross_self_att)

            # =========================
            # 🔹 FEED FORWARD (Pre-LN)
            # =========================
            norm_cross_out = self.layer_norm(
                cross_out,
                self.W_norm_ff_a[layer], self.W_norm_ff_b[layer]
            )

            ff_layer_1 = F.relu(norm_cross_out @ self.W_ff1[layer].weight.T +self.W_ff1[layer].bias)
            ff_layer_1= self.dropout['ff_layer_1'].train(self.is_training).forward(ff_layer_1)
            ff_output = ff_layer_1 @ self.W_ff2[layer].weight.T + self.W_ff2[layer].bias

            

            # =========================
            # 🔹 STORE ACTIVATIONS
            # =========================

            # residuals
            self.residual_self.append(inputs)
            self.residual_cross.append(self_out)
            self.residual_ff.append(cross_out)

            # FF
            self.ff_output.append(ff_output)
            self.ff_layer_1_output.append(ff_layer_1)
            self.ff_inputs.append(norm_cross_out)  # PRE-LN input

            # Cross Attention
            self.cross_att_output.append(cross_self_att)
            self.cross_att_inputs.append(norm_self_out)  # PRE-LN input
            self.cross_att_o.append(cross_att_o)
            self.cross_att_q.append(cross_att_Q)
            self.cross_att_k.append(cross_att_K)
            self.cross_att_v.append(cross_att_V)
            self.cross_att_a.append(cross_att_A)
            self.cross_att_raw_a.append(cross_att_raw_A)

            # Self Attention
            self.self_att_output.append(multi_self_att)
            self.self_att_inputs.append(norm_inputs)  # PRE-LN input
            self.self_att_o.append(multi_self_att_o)
            self.self_att_q.append(multi_self_att_Q)
            self.self_att_k.append(multi_self_att_K)
            self.self_att_v.append(multi_self_att_V)
            self.self_att_a.append(multi_self_att_A)
            self.self_att_raw_a.append(multi_self_att_raw_A)
            # total = self.self_att_a[layer].numel()
            # count = (self.self_att_a[layer] > 1).sum().item()

            # print("count:", count)
            # print("percentage:", count / total)
            inputs = cross_out + self.dropout['ff_out'].train(self.is_training).forward(ff_output)
        self.H = inputs
        # print(self.W_voc.bias.shape)
        logits = inputs @ self.emb.weight.T
        logits = logits
        prob = F.softmax(logits, dim=2)
        self.prob = prob
        # print("logits max:", logits.abs().max().item())
        # print("logits std:", logits.std().item())
        # print("prob max:", prob.max().item())
        # print("prob entropy:", -(prob * (prob + 1e-9).log()).sum(dim=-1).mean().item())

# Healthy values:
# logits std: 1-5
# prob max: 0.3-0.7
# entropy: reasonably high

# Overconfident:
# logits std: >10
# prob max: >0.99
# entropy: near zero
        return prob


    def back_pre(self, targets, prob, smoothing=0.1):


        # rescaled_targets = np.expand_dims(targets, axis=-1)
        # targets_t = torch.tensor(rescaled_targets, dtype=torch.long, device=device)
        # # print(targets_t.shape)
        # mask = (targets_t != 0)
        log_prob = torch.log(prob.clamp(min=1e-9))   # your current approach (less stable)

        targets_t = torch.tensor(targets, dtype=torch.long, device=device).unsqueeze(-1)
        mask = (targets_t != 0)

        nll_loss    = -log_prob.gather(dim=-1, index=targets_t)       # (B, T, 1)
        smooth_loss = -log_prob.mean(dim=-1, keepdim=True)             # (B, T, 1)
        
        loss_per_token = (1 - smoothing) * nll_loss + smoothing * smooth_loss
        loss = (loss_per_token * mask).sum() / mask.sum()
        # print(loss)
        delta_z = self.gradient_softmax_cross_entropy(targets, prob, smoothing)

        # apply mask and normalise — same normalisation as loss
        delta_z = delta_z * mask / mask.sum()
        delta_W_voc = self.gradient_W_voc(delta_z).T
        # print(delta_z.shape)
        # delta_W_voc_b= delta_z.sum(dim=(0,1))
        # delta_H = delta_z @ self.emb.weight
    
        delta_H = delta_z@ self.emb.weight
        delta_H= delta_H
        # print("decoder ",delta_H.abs().mean())A
         # check= self.W_voc.clone()
        # W_voc_grad=self.W_voc_ada.grad(delta_W_voc,self.W_voc.weight)
        # self.emb.weight -=W_voc_grad
        # print(W_voc_grad.abs().mean())
        # self.W_voc.weight-= self.W_voc_ada.grad(delta_W_voc,self.W_voc.bias)
        # self.W_voc.bias-= self.W_voc_ada.grad(delta_W_voc_b,self.W_voc.bias)
        # print(self.W_voc.weight.abs().mean(),W_voc_grad.abs().mean())
            # print(delta_W_voc.abs().mean())
            # print(self.W_voc.weight.abs().mean(),delta_W_voc.abs().mean())
            # self.W_voc -= delta_W_voc
            # print(self.W_voc.weight.abs().mean(),delta_W_voc.abs().mean())
            # self.W_voc.weight -= 0.0001*delta_W_voc
        

        prev_delta = delta_H
        # print(prev_delta.abs().mean())
        del_E = torch.zeros((self.batch_size, self.sent_len, self.d_model),device=device) 
        for layer in range(self.layers - 1, -1, -1):
            # print(f"layer {layer}")
            # =========================
            # 🔹 FF BLOCK
            # y = cross_out + FF(LN(cross_out))
            # =========================

            cross_out = self.residual_ff[layer]

            # residual path
            delta_residual = prev_delta

            # FF path
            delta_ff = prev_delta

            # FF grads
            ff_w1_grad, ff_b1_grad, ff_w2_grad, ff_b2_grad, ff_x_grad = self.grad_ff(
                self.dropout['ff_out'].backward(delta_ff),
                self.ff_layer_1_output[layer],
                self.W_ff2[layer].weight.T,
                self.W_ff1[layer].weight.T,
                self.ff_inputs[layer]
            )
            # print(ff_w1_grad.abs().mean(),self.W_ff1[layer].abs().mean())
            # back through LN (input was cross_out)
            delta_ln_ff, ff_alpha_grad, ff_beta_grad = self.grad_layer_norm_pre(
                ff_x_grad,
                self.W_norm_ff_a[layer],
                self.W_norm_ff_b[layer],
                cross_out
            )
            # print(delta_ln_ff.abs().mean(),delta_residual.abs().mean())

            # combine
            delta_cross_out = delta_residual + delta_ln_ff

            # =========================
            # 🔹 CROSS ATTENTION BLOCK
            # y = self_out + CrossAtt(LN(self_out))
            # =========================

            self_out = self.residual_cross[layer]

            delta_residual = delta_cross_out
            delta_cross = delta_cross_out

            delta_E_k, delta_E_v, delta_x, delta_W_q, delta_W_k, delta_W_v, delta_W_o,delta_W_q_b, delta_W_k_b, delta_W_v_b, delta_W_o_b = self.grad_cross_att(
                self.dropout['cross_att_out'].backward(delta_cross),
                self.W_cross_o[layer].weight,
                self.cross_att_o[layer],
                self.cross_att_a[layer],
                self.cross_att_raw_a[layer],
                self.cross_att_v[layer],
                self.cross_att_q[layer],
                self.cross_att_k[layer],
                self.cross_att_inputs[layer],
                self.E,
                self.W_cross_q[layer].weight,
                self.W_cross_k[layer].weight,
                self.W_cross_v[layer].weight
            )
            # print(delta_x.abs().mean())
            # back through LN
            delta_ln_cross, cross_alpha_grad, cross_beta_grad = self.grad_layer_norm_pre(
                delta_x,
                self.W_norm_cross_a[layer],
                self.W_norm_cross_b[layer],
                self_out
            )
            # print(delta_ln_cross.abs().mean())
            delta_self_out = delta_residual + delta_ln_cross
            del_E += (delta_E_k + delta_E_v)

            # =========================
            # 🔹 SELF ATTENTION BLOCK
            # y = x + SelfAtt(LN(x))
            # =========================

            x_input = self.self_att_inputs[layer]  # LN input stored

            delta_residual = delta_self_out
            delta_self = delta_self_out

            self_delta_X_k, self_delta_X_v, self_delta_X_q, \
            self_delta_W_q, self_delta_W_k, self_delta_W_v, self_delta_W_o, self_delta_W_q_b, self_delta_W_k_b, self_delta_W_v_b, self_delta_W_o_b = self.grad_multi_self_att(
                self.dropout['self_att_out'].backward(delta_self),
                self.W_o[layer].weight,
                self.self_att_o[layer],
                self.self_att_a[layer],
                self.self_att_raw_a[layer],
                self.self_att_v[layer],
                self.self_att_q[layer],
                self.self_att_k[layer],
                self.self_att_inputs[layer],
                self.W_q[layer].weight,
                self.W_k[layer].weight,
                self.W_v[layer].weight
            )

            delta_att = self_delta_X_v + self_delta_X_k + self_delta_X_q
            # print(delta_att.abs().mean())
            # back through LN
            delta_ln_self, self_alpha_grad, self_beta_grad = self.grad_layer_norm_pre(
                delta_att,
                self.W_norm_self_a[layer],
                self.W_norm_self_b[layer],
                x_input
            )

            prev_delta = delta_residual + delta_ln_self
            # print(prev_delta.abs().mean())
            # =========================
            # 🔹 WEIGHT UPDATES
            # =========================
            with torch.no_grad():
                # FF
                self.W_ff1[layer].weight -= self.W_ff1_ada[layer].grad(ff_w1_grad,self.W_ff1[layer].weight)
                self.W_ff2[layer].weight -= self.W_ff2_ada[layer].grad(ff_w2_grad,self.W_ff2[layer].weight)
                # print((self.W_ff1[layer].weight**2).mean().sqrt(),self.W_ff1_ada[layer].grad(ff_w1_grad,self.W_ff1[layer].weight).abs().mean())
                # print(self.W_ff2[layer].weight.abs().mean(),self.W_ff2_ada[layer].grad(ff_w2_grad,self.W_ff1[layer].weight).abs().mean())
                self.W_ff1[layer].bias -= self.W_ff1_ada[layer].grad(ff_b1_grad,self.W_ff1[layer].bias)
                self.W_ff2[layer].bias-= self.W_ff2_ada[layer].grad(ff_b2_grad,self.W_ff2[layer].bias)
                # print(self.b_ff1[layer].abs().mean(),self.b_ff1_ada[layer].grad(ff_b1_grad).abs().mean())
                # print(self.b_ff2[layer].abs().mean(),self.b_ff2_ada[layer].grad(ff_b2_grad).abs().mean())
                # print(self.W_ff1[layer].weight.abs().mean(),self.W_ff1_ada[layer].grad(ff_w1_grad).abs().mean())

                # # Cross
                self.W_cross_q[layer].weight -= self.W_cross_q_ada[layer].grad(delta_W_q,self.W_cross_q[layer].weight)
                self.W_cross_k[layer].weight -= self.W_cross_k_ada[layer].grad(delta_W_k,self.W_cross_k[layer].weight)
                self.W_cross_v[layer].weight -= self.W_cross_v_ada[layer].grad(delta_W_v,self.W_cross_v[layer].weight)
                self.W_cross_o[layer].weight -= self.W_cross_o_ada[layer].grad(delta_W_o,self.W_cross_o[layer].weight)
                self.W_cross_q[layer].bias -= self.W_cross_q_ada[layer].grad(delta_W_q_b,self.W_cross_q[layer].bias)
                self.W_cross_k[layer].bias -= self.W_cross_k_ada[layer].grad(delta_W_k_b,self.W_cross_k[layer].bias)
                self.W_cross_v[layer].bias -= self.W_cross_v_ada[layer].grad(delta_W_v_b,self.W_cross_v[layer].bias)
                self.W_cross_o[layer].bias -= self.W_cross_o_ada[layer].grad(delta_W_o_b,self.W_cross_o[layer].bias)
                # print(self.W_cross_q[layer].weight.abs().mean(),self.W_cross_q_ada[layer].grad(delta_W_q,self.W_cross_q[layer].weight).abs().mean())
                # print(self.W_cross_k[layer].weight.abs().mean(),self.W_cross_k_ada[layer].grad(delta_W_k).abs().mean())
                # print(self.W_cross_v[layer].weight.abs().mean(),self.W_cross_v_ada[layer].grad(delta_W_v).abs().mean())
                # print(self.W_cross_o[layer].weight.abs().mean(),self.W_cross_o_ada[layer].grad(delta_W_o).abs().mean())
                # # Self
                self.W_q[layer].weight -= self.W_q_ada[layer].grad(self_delta_W_q,self.W_q[layer].weight)
                self.W_k[layer].weight -= self.W_k_ada[layer].grad(self_delta_W_k,self.W_k[layer].weight)
                self.W_v[layer].weight -= self.W_v_ada[layer].grad(self_delta_W_v,self.W_v[layer].weight)
                self.W_o[layer].weight -= self.W_o_ada[layer].grad(self_delta_W_o,self.W_o[layer].weight)
                self.W_q[layer].bias -= self.W_q_ada[layer].grad(self_delta_W_q_b,self.W_q[layer].bias)
                self.W_k[layer].bias -= self.W_k_ada[layer].grad(self_delta_W_k_b,self.W_k[layer].bias)
                self.W_v[layer].bias -= self.W_v_ada[layer].grad(self_delta_W_v_b,self.W_v[layer].bias)
                self.W_o[layer].bias -= self.W_o_ada[layer].grad(self_delta_W_o_b,self.W_o[layer].bias)
                # print(self.W_q[layer].weight.abs().mean(),self.W_q_ada[layer].grad(self_delta_W_q).abs().mean())
                # print(self.W_k[layer].weight.abs().mean(),self.W_k_ada[layer].grad(self_delta_W_k).abs().mean())
                # print(self.W_v[layer].weight.abs().mean(),self.W_v_ada[layer].grad(self_delta_W_v).abs().mean())
                # print(self.W_o[layer].weight.abs().mean(),self.W_o_ada[layer].grad(self_delta_W_o).abs().mean())
                # # # Norms
                self.W_norm_self_a[layer] -= self.W_norm_self_a_ada[layer].grad(self_alpha_grad,self.W_norm_self_a[layer])
                self.W_norm_self_b[layer] -= self.W_norm_self_b_ada[layer].grad(self_beta_grad,self.W_norm_self_b[layer])
                # print(abs_mean(self.W_norm_self_a[layer]),abs_mean(self.W_norm_self_a_ada[layer].grad(self_alpha_grad)))
                # print(abs_mean(self.W_norm_self_b[layer]),abs_mean(self.W_norm_self_b_ada[layer].grad(self_beta_grad)))
                self.W_norm_cross_a[layer] -= self.W_norm_cross_a_ada[layer].grad(cross_alpha_grad,self.W_norm_cross_a[layer])
                self.W_norm_cross_b[layer] -= self.W_norm_cross_b_ada[layer].grad(cross_beta_grad,self.W_norm_cross_b[layer])
                # print(abs_mean(self.W_norm_cross_a[layer]),abs_mean(self.W_norm_cross_a_ada[layer].grad(cross_alpha_grad)))
                # print(abs_mean(self.W_norm_cross_b[layer]),abs_mean(self.W_norm_cross_b_ada[layer].grad(cross_beta_grad)))
                self.W_norm_ff_a[layer] -= self.W_norm_ff_a_ada[layer].grad(ff_alpha_grad,self.W_norm_ff_a[layer])
                self.W_norm_ff_b[layer] -= self.W_norm_ff_b_ada[layer].grad(ff_beta_grad,self.W_norm_ff_b[layer] )
                # print(abs_mean(self.W_norm_ff_a[layer] ),abs_mean(self.W_norm_ff_a_ada[layer].grad(ff_alpha_grad)))
                # print(abs_mean(self.W_norm_ff_b[layer] ),abs_mean(self.W_norm_ff_b_ada[layer].grad(ff_beta_grad)))
                pass
        # print(prev_delta)
        prev_delta= self.dropout['emb'].backward(prev_delta)
        flat_tokens = torch.tensor(self.D, device=device).view(-1)          # (B*T,)
        flat_delta  = prev_delta.view(-1, self.d_model)                      # (B*T, d_model)
        emb_grad  = torch.zeros(self.voc_size, self.d_model, device=device)
        emb_grad.index_add_(0, flat_tokens, flat_delta)
        
        # counts = torch.zeros(self.voc_size, device=device)
        # counts.index_add_(0, flat_tokens, torch.ones(flat_tokens.shape[0], device=device))
        # counts = counts.clamp(min=1)
        # W_voc_grad = W_voc_grad / counts.unsqueeze(1)
        # rows_list   = flat_tokens.unique().tolist()
        # emb_grad = self.emb_ad.grad_emb(W_voc_grad[rows_list], rows_list,self.emb.weight[rows_list])
        # pos_grad = self.pos_ada.grad(torch.sum(prev_delta, dim=0),self.pos.weight)
        delta_W_voc+= emb_grad
        # self.emb.weight -= self.W_voc_ada.grad(delta_W_voc,self.emb.weight) 
        # self.pos.weight -= pos_grad
        # print(self.emb.weight.abs().mean())
        
        # FIX: release all cached activation tensors and per-batch state
        #      so iteration N+1 starts with exactly the same memory footprint as iteration N.
        # self.clear_memory()
        # At the end of back_pre, before return:
        # print(len(self.self_att_a), len(self.cross_att_a))
        return del_E,loss,delta_W_voc,torch.sum(prev_delta, dim=0)

    # ─────────────────────────────────────────────
    # Gradient helpers
    # ─────────────────────────────────────────────
    def grad_layer_norm_pre(self, delta, alpha, beta, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.epsilon)

        x_hat = (x - mean) / std

        del_alpha = (delta * x_hat).sum(dim=(0, 1))
        del_beta = delta.sum(dim=(0, 1))

        delta_x_hat = delta * alpha

        del_x = (1.0 / std) * (
            delta_x_hat
            - delta_x_hat.mean(dim=-1, keepdim=True)
            - x_hat * (delta_x_hat * x_hat).mean(dim=-1, keepdim=True)
        )

        return del_x, del_alpha, del_beta
    def gradient_softmax_cross_entropy(self, targets, prob, smoothing=0.1):
        targets = torch.tensor(targets, device=device)
        B, T, V = prob.shape
        eps = smoothing
        delta_z = prob.clone()

        delta_z[torch.arange(B)[:,None], torch.arange(T), targets] -= 1.0
        # delta_z[torch.arange(B)[:, None], torch.arange(T), targets] -= (1.0 - smoothing)
        delta_z = (1 - eps) * delta_z + eps * (prob - 1.0 / V)
        return delta_z

        # subtract smoothing/V from every vocab position
        # this is the gradient of the uniform distribution penalty term
        # mathematically: d/dz [ -eps * mean(log p) ] = eps/V * (p - 1/p... )
        # simplified: just subtract eps/V from every position since d(-log p_i)/dz_j = p_j - 1_{i==j}
        # summed uniformly: p_j - 1/V  →  already captured by subtracting 1/V
                                   # (B, T, V)

    def gradient_W_voc(self, delta_z):
    #    return torch.einsum('bti,btj->ij', self.H, delta_z)
         return (self.H.view(-1, self.d_model).T @ delta_z.view(-1, self.voc_size))

    def grad_layer_norm(self, delta, alpha, beta, x, f_x, r_):
     combined = (x + f_x)
     mean  = combined.mean(dim=-1, keepdim=True)
     std   = torch.sqrt(combined.var(dim=-1, keepdim=True) + self.epsilon)
     x_hat = (combined - mean) / std

     del_alpha = (delta * x_hat).sum(dim=(0, 1))
     del_beta  = delta.sum(dim=(0, 1))
     delta_x_hat = delta * alpha
     
     del_x = (1.0 / std) * (
          delta_x_hat
          - delta_x_hat.mean(dim=-1, keepdim=True)
          - x_hat * (delta_x_hat * x_hat).mean(dim=-1, keepdim=True)
     )
    
     # FIX 5: free temporaries before returning
     return del_x, del_alpha, del_beta

    def grad_ff(self, delta, layer_1_output, ff_w_2, ff_w_1, x):
        # ff_w2 gradient: sum over batch of (layer_1_output[B].T @ delta[B])
        # Old: for B in range(self.batch_size): delta_ff_w2 += layer_1_output[B].T @ delta[B]
        # delta_ff_w2 = torch.einsum('bti,btj->ij', layer_1_output, delta).T
        delta_ff_w2 = (layer_1_output.view(-1, self.d_ff).T @ delta.view(-1, self.d_model)).T
        delta_ff_b2 = delta.sum(dim=(0, 1))
        
        # Backprop through W_ff2
        delta_h = delta @ ff_w_2.T
        
        # ReLU backward
        delta_h= self.dropout['ff_layer_1'].backward(delta_h)
        delta_a = delta_h * (layer_1_output > 0).float()
        
        delta_ff_b1 = delta_a.sum(dim=(0, 1))
        
        # ff_w1 gradient: sum over batch of (x[B].T @ delta_a[B])
        # Old: for B in range(self.batch_size): delta_ff_w1 += x[B].T @ delta_a[B]
        # delta_ff_w1 = torch.einsum('bti,btj->ij', x, delta_a).T
        delta_ff_w1= (x.view(-1, self.d_model).T @ delta_a.view(-1, self.d_ff)).T
        
        delta_x = delta_a @ ff_w_1.T
        
        return delta_ff_w1, delta_ff_b1, delta_ff_w2, delta_ff_b2, delta_x

    def grad_cross_att(self, delta, W_o, O, A,raw_A, V, Q, K, X, E, W_q, W_k, W_v):
        B, T, _ = delta.shape
        delta_o = delta @ W_o.T
        # delta_W_o = torch.einsum('bti,btj->ij', O, delta)
        delta_W_o=(O.view(-1, self.d_model).T @ delta.view(-1, self.d_model))
        delta_W_o_b= delta.sum(dim=(0,1))
        heads_delta_o = delta_o.view(B, T, self.h_count, self.d_k).transpose(1, 2)
        # print(heads_delta_o.shape,V.shape)
        heads_delta_A= heads_delta_o@V.transpose(-2,-1)
        heads_delta_A= self.dropout['cross_att_a'].backward(heads_delta_A)
        heads_delta_V= A.transpose(-2,-1)@heads_delta_o
        dot_delA_A= (heads_delta_A * raw_A).sum(dim=-1, keepdim=True)
        heads_delta_S = raw_A * (heads_delta_A - dot_delA_A)
        heads_delta_Q= (heads_delta_S @ K) / np.sqrt(self.d_k)
        heads_delta_K= (heads_delta_S.transpose(-2, -1) @ Q) / np.sqrt(self.d_k)
        heads_delta_V= heads_delta_V.transpose(1, 2).contiguous().view(B, T, -1)   
        heads_delta_Q= heads_delta_Q.transpose(1, 2).contiguous().view(B, T, -1)   
        heads_delta_K= heads_delta_K.transpose(1,2).contiguous().view(B, T, -1)  
        delta_E_K = heads_delta_K @ W_k.T
        delta_E_V = heads_delta_V @ W_v.T
        delta_X_Q = heads_delta_Q @ W_q.T
        # delta_W_q= torch.einsum('bti,btj->ij', X, heads_delta_Q)
        # delta_W_k = torch.einsum('bti,btj->ij', E, heads_delta_K)
        # delta_W_v = torch.einsum('bti,btj->ij', E, heads_delta_V)
        delta_W_q=(X.view(-1, self.d_model).T @ heads_delta_Q.view(-1, self.d_model))
        delta_W_k=(E.view(-1, self.d_model).T @ heads_delta_K.view(-1, self.d_model))
        delta_W_v=(E.view(-1, self.d_model).T @ heads_delta_V.view(-1, self.d_model))
        delta_W_q_b=  heads_delta_Q.sum(dim=(0,1))
        delta_W_k_b = heads_delta_K.sum(dim=(0,1))
        delta_W_v_b=  heads_delta_V.sum(dim=(0,1))
        return delta_E_K, delta_E_V, delta_X_Q, delta_W_q, delta_W_k, delta_W_v, delta_W_o,delta_W_q_b,delta_W_k_b,delta_W_v_b,delta_W_o_b

    def grad_multi_self_att(self, delta, W_o, O, A,raw_A, V, Q, K, X, W_q, W_k, W_v):
        B, T, _ = delta.shape
        delta_o = delta @ W_o.T
        # delta_W_o = torch.einsum('bti,btj->ij', O, delta)
        delta_W_o=(O.view(-1, self.d_model).T @ delta.view(-1, self.d_model))
        delta_W_o_b= delta.sum(dim=(0,1))
        heads_delta_o = delta_o.view(B, T, self.h_count, self.d_k).transpose(1, 2)
        # print(heads_delta_o.shape,V.shape)
        heads_delta_A= heads_delta_o@V.transpose(-2,-1)
        heads_delta_A= self.dropout['self_att_a'].backward(heads_delta_A)
        heads_delta_V= A.transpose(-2,-1)@heads_delta_o
        dot_delA_A= (heads_delta_A * raw_A).sum(dim=-1, keepdim=True)
        heads_delta_S = raw_A * (heads_delta_A - dot_delA_A)
        heads_delta_Q= (heads_delta_S @ K) / np.sqrt(self.d_k)
        heads_delta_K= (heads_delta_S.transpose(-2, -1) @ Q) / np.sqrt(self.d_k)
        heads_delta_V= heads_delta_V.transpose(1, 2).contiguous().view(B, T, -1)   
        heads_delta_Q= heads_delta_Q.transpose(1, 2).contiguous().view(B, T, -1)   
        heads_delta_K= heads_delta_K.transpose(1,2).contiguous().view(B, T, -1)  
        delta_X_K = heads_delta_K @ W_k.T
        delta_X_V = heads_delta_V @ W_v.T
        delta_X_Q = heads_delta_Q @ W_q.T
        # delta_W_q= torch.einsum('bti,btj->ij', X, heads_delta_Q)
        # delta_W_k = torch.einsum('bti,btj->ij', X, heads_delta_K)
        # delta_W_v = torch.einsum('bti,btj->ij', X, heads_delta_V)
        delta_W_q=(X.view(-1, self.d_model).T @ heads_delta_Q.view(-1, self.d_model))
        delta_W_k=(X.view(-1, self.d_model).T @ heads_delta_K.view(-1, self.d_model))
        delta_W_v=(X.view(-1, self.d_model).T @ heads_delta_V.view(-1, self.d_model))
        delta_W_q_b=  heads_delta_Q.sum(dim=(0,1))
        delta_W_k_b = heads_delta_K.sum(dim=(0,1))
        delta_W_v_b=  heads_delta_V.sum(dim=(0,1))
   
        return delta_X_K, delta_X_V, delta_X_Q, delta_W_q, delta_W_k, delta_W_v, delta_W_o,delta_W_q_b,delta_W_k_b,delta_W_v_b,delta_W_o_b

    # ─────────────────────────────────────────────
    # Attention & norm
    # ─────────────────────────────────────────────
    def masked_self_att(self, x, layer):
        B, T, _ = x.shape
        Q = (x @ self.W_q[layer].weight+self.W_q[layer].bias).view(B, T, self.h_count, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = (x @ self.W_k[layer].weight+self.W_k[layer].bias).view(B, T, self.h_count, self.d_k).transpose(1, 2)
        V = (x @ self.W_v[layer].weight+self.W_v[layer].bias).view(B, T, self.h_count, self.d_k).transpose(1, 2)
        temp=1
        S = Q @ K.transpose(-2, -1) /(math.sqrt(self.d_k)*temp)  
        pad_mask = torch.tensor(self.pad_mask_k|self.pad_mask_q, dtype=torch.bool, device=device)    # (B, H, T, T)
        S = S + self.causal_mask[:T, :T]
        S= S.masked_fill(pad_mask,negative_inf)
    
        A = F.softmax(S, dim=-1)  
        A = torch.nan_to_num(A, nan=0.0)
        raw_A= A
        A= self.dropout['self_att_a'].train(self.is_training).forward(A)                                  # (B, H, T, T)
        O = (A @ V).transpose(1, 2).contiguous().view(B, T, -1)     # (B, T, d_model)
        output = O @ self.W_o[layer].weight +self.W_o[layer].bias
        return output, O, Q, K, V, A,raw_A

    def cross_attention(self, z1, E, layer,E_pad_mask):
        B, T, _ = z1.shape
        E_B,E_T,_= E.shape
        Wq = self.W_cross_q[layer].weight
        Wk = self.W_cross_k[layer].weight
        Wv = self.W_cross_v[layer].weight
        Wq_b = self.W_cross_q[layer].bias
        Wk_b = self.W_cross_k[layer].bias
        Wv_b = self.W_cross_v[layer].bias
        # print(Wk.shape)
        Q = (z1 @ Wq+Wq_b).view(B, T, self.h_count, self.d_k).transpose(1, 2)
        K = (E @ Wk+Wk_b).view(E_B, E_T, self.h_count, self.d_k).transpose(1, 2)
        V = (E @ Wv+Wv_b).view(E_B, E_T, self.h_count, self.d_k).transpose(1, 2)
        temp=1
        S = Q @ K.transpose(-2, -1) / (math.sqrt(self.d_k)*temp)        # (B, H, T, T)
        # S = S - S.max(dim=-1, keepdim=True)[0]
        pad_mask = torch.tensor(E_pad_mask|self.pad_mask_q, dtype=torch.bool, device=device) 
        S= S.masked_fill(pad_mask,negative_inf)
        A = F.softmax(S, dim=-1)  
        A = torch.nan_to_num(A, nan=0.0)
        
        raw_A= A   
        A= self.dropout['cross_att_a'].train(self.is_training).forward(A)                               # (B, H, T, T)
        O = (A @ V).transpose(1, 2).contiguous().view(B, T, -1)
        output = O @ self.W_cross_o[layer].weight+self.W_cross_o[layer].bias

        return output, O, Q, K, V,  A,raw_A

    def layer_norm(self, X, alpha, beta):
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.epsilon)
        return ((X - mean) / std) * alpha + beta
    def create_pad_mask_q(self, X):
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, device=device)
            pad_q = (X == 0).unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
            # pad_k = (X == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            return pad_q  
    def create_pad_mask_k(self, X):
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, device=device)
            # pad_q = (X == 0).unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
            pad_k = (X == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            return  pad_k  # (B, 1, T, T)
    def get_sinusoidal_positional_encoding(self,seq_len, d_model):
        pos = torch.arange(seq_len).unsqueeze(1)
        i = torch.arange(d_model).unsqueeze(0)

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angles = pos * angle_rates

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])

        return pe.to(device)
    # ─────────────────────────────────────────────
    # Memory management
    # ─────────────────────────────────────────────
    def clear_memory(self):
        """Release all activation caches and per-batch state tensors."""
        self.ff_output.clear()
        self.ff_inputs.clear()
        self.ff_layer_1_output.clear()

        self.cross_att_output.clear()
        self.cross_att_inputs.clear()
        self.cross_att_o.clear()
        self.cross_att_q.clear()
        self.cross_att_k.clear()
        self.cross_att_v.clear()
        self.cross_att_s.clear()
        self.cross_att_a.clear()
        self.cross_att_raw_a.clear()

        self.self_att_output.clear()
        self.self_att_inputs.clear()
        self.self_att_o.clear()
        self.self_att_q.clear()
        self.self_att_k.clear()
        self.self_att_v.clear()
        self.self_att_s.clear()
        self.self_att_a.clear()
        self.self_att_raw_a.clear()
        self.dropout['emb'].clear()
        self.dropout['self_att_a'].clear()
        self.dropout['self_att_out'].clear()
        self.dropout['cross_att_a'].clear()
        self.dropout['cross_att_out'].clear()
        self.dropout['ff_layer_1'].clear()
        self.dropout['ff_out'].clear()
        
        # FIX: also null out per-batch state so GC can free these tensors
        self.H = None
        self.prob = None
        self.E = None
        self.D = None
    def val(self):
        self.is_training= False
    def train(self):
        self.is_training= True

# ─────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────
def log_grad_flow(decoder):
    print("\n=== Gradient Flow ===")

    for i in range(decoder.layers):
        print(f"\nLayer {i}")

        # FF
        print("FF1 grad:", decoder.W_ff1[i].weight.abs().mean().item())
        print("FF2 grad:", decoder.W_ff2[i].weight.abs().mean().item())

        # Self Attention
        print("Self Q:", decoder.W_q[i].weight.abs().mean().item())
        print("Self K:", decoder.W_k[i].weight.abs().mean().item())
        print("Self V:", decoder.W_v[i].weight.abs().mean().item())
        print("Self O:", decoder.W_o[i].weight.abs().mean().item())

        # Cross Attention
        print("Cross Q:", decoder.W_cross_q[i].weight.abs().mean().item())
        print("Cross K:", decoder.W_cross_k[i].weight.abs().mean().item())
        print("Cross V:", decoder.W_cross_v[i].weight.abs().mean().item())
        print("Cross O:", decoder.W_cross_o[i].weight.abs().mean().item())

        # LayerNorm
        print("Norm self alpha:", decoder.W_norm_self_a[i].abs().mean().item())
        print("Norm cross alpha:", decoder.W_norm_cross_a[i].abs().mean().item())
        print("Norm ff alpha:", decoder.W_norm_ff_a[i].abs().mean().item())
def predict(y,target, encoder, decoder):
    
    y = torch.tensor(y).to(device)

    E,E_pad_mask = encoder.fit_pre(y)

    start = torch.tensor([[2]], device=device)

    print(tokenizer.decode(target[0].tolist()))
    
    
    i = 0
    while start[0, -1] != 3:
        prob = decoder.fit_pre(start, E,E_pad_mask)

        token_prob, index = torch.max(prob, dim=-1)

        next_token = index[0, i].item()

        print(tokenizer.decode([next_token]), end=" ")

        start = torch.cat(
            [start, torch.tensor([[next_token]], device=device)], dim=1
        )

        i += 1
        
def calculate_validation_loss(encoder, decoder, x_val_encoder, x_val_decoder, x_val_target, batch_size=64):
    """
    Calculate validation loss without updating any weights.
    
    Args:
        encoder: trained Encoder instance
        decoder: trained Decoder instance
        x_val_encoder: numpy array of encoder inputs
        x_val_decoder: numpy array of decoder inputs  
        x_val_target:  numpy array of target tokens
        batch_size:    must match the batch_size the models were built with
    
    Returns:
        avg_loss (float): mean cross-entropy loss over the validation set
    """
    total_loss = 0.0
    num_batches = 0
    encoder.val()
    decoder.val()
    with torch.no_grad():
        for i in range(0, len(x_val_encoder) - batch_size, batch_size):
            enc_batch = x_val_encoder[i : i + batch_size]
            dec_batch = x_val_decoder[i : i + batch_size]
            tgt_batch = x_val_target[i : i + batch_size]

            # ---------- forward passes ----------
            E,E_pad_mask    = encoder.fit_pre(enc_batch)
            prob = decoder.fit_pre(dec_batch, E,E_pad_mask)

            # ---------- loss (mirrors your back_pre logic) ----------
            rescaled_targets = np.expand_dims(tgt_batch, axis=-1)
            targets_t = torch.tensor(rescaled_targets, dtype=torch.long, device=device)
            # print(targets_t.shape)
            tokens_prob = prob.gather(dim=-1, index=targets_t)
            tokens_prob = torch.clamp(tokens_prob, min=1e-9)
            log_tokens_prob = torch.log(tokens_prob)
            # print(torch.tensor(targets).shape,log_tokens_prob.shape)
            mask= (targets_t!=0)
        
            loss = (-torch.sum(log_tokens_prob*mask))/mask.sum()

            total_loss  += loss.item()
            num_batches += 1

            # free activations so memory stays flat across batches
            encoder.clear_memory()
            decoder.clear_memory()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss
def most_similar(word_idx, emb,topk=10):
    v = emb[word_idx]                     # (d,)
    sims = F.cosine_similarity(v.unsqueeze(0), emb)  # (vocab,)
    vals, idx = torch.topk(sims, topk+1)   # +1 to skip itself
    return tokenizer.decode(idx[1:].tolist()), vals[1:]
