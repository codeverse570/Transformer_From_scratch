import gc
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from encoder import Encoder
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
torch.set_float32_matmul_precision('high')


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
        self.dropout= {'emb':Dropout(),'enc_out':Dropout(),'self_att_a':Dropout(),'self_att_out':Dropout(),'cross_att_a':Dropout(),'cross_att_out':Dropout(),'ff_layer_1':Dropout(),'ff_out':Dropout()}
        self.schedular=schedular
        self.D = []
        
        self.pad_mask=[]
        self.grads=[]
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
        self.L=[]

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
        self._del_E = torch.zeros(batch_size, sent_len, d_model, device=device)
        self.grad_ff        = torch.compile(self.grad_ff, backend='inductor' , fullgraph=False)
        self.grad_cross_att = torch.compile(self.grad_cross_att,backend='inductor', fullgraph=False)
        self.grad_multi_self_att = torch.compile(self.grad_multi_self_att, backend='inductor',fullgraph=False)
        self.grad_layer_norm_pre = torch.compile(self.grad_layer_norm_pre, backend='inductor',fullgraph=False)
        self.delta_H    = torch.zeros(batch_size, sent_len, d_model, device=device)
        self.delta_z    = torch.zeros(batch_size, sent_len, voc_size, device=device)
        self.emb_grad   = torch.zeros(voc_size, d_model, device=device)
        self.W_qkv = nn.ModuleList([
            nn.Linear(d_model, 3 * d_model, device=device) 
            for _ in range(layers)
        ])
        # self.grad_cross_att = torch.compile(self.grad_cross_att, backend='aot_eager')
        # self.masked_self_att  = torch.compile(self.masked_self_att,   backend='aot_eager')
        # self.grad_ff= torch.compile(self.grad_ff,backend='aot_eager')
        # self.layer_norm = torch.compile(self.layer_norm, fullgraph=True)
        # self._compiled_ff = torch.compile(self._ff_forward, fullgraph=True)
        

    # ─────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove compiled functions - they can't be pickled
        for key in ['grad_ff', 'grad_cross_att', 'grad_multi_self_att', 'grad_layer_norm_pre']:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recompile after loading
        self.grad_ff             = torch.compile(self.grad_ff,             backend='inductor', fullgraph=False)
        self.grad_cross_att      = torch.compile(self.grad_cross_att,      backend='inductor', fullgraph=False)
        self.grad_multi_self_att = torch.compile(self.grad_multi_self_att, backend='inductor', fullgraph=False)
        self.grad_layer_norm_pre = torch.compile(self.grad_layer_norm_pre, backend='inductor', fullgraph=False)
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
        W_q = nn.ModuleList()
        W_k = nn.ModuleList()
        W_v = nn.ModuleList()
        W_o = nn.ModuleList()
        W_cross_q = nn.ModuleList()
        W_cross_k = nn.ModuleList()
        W_cross_v = nn.ModuleList()
        W_cross_o = nn.ModuleList()
        W_ff1 = nn.ModuleList()
        W_ff2 = nn.ModuleList()
        W_norm_self_a  = nn.ParameterList()
        W_norm_self_b  = nn.ParameterList()
        W_norm_cross_a = nn.ParameterList()
        W_norm_cross_b = nn.ParameterList()
        W_norm_ff_a    = nn.ParameterList()
        W_norm_ff_b    = nn.ParameterList()

        std_qk      = 1.0 / math.sqrt(self.d_model)
        std_general = 0.02
        res_scale   = 1.0 / math.sqrt(2 * self.layers)

        for i in range(self.layers):
            # ── Attention projections ──────────────────────────
            Wq  = nn.Linear(self.d_model, self.d_model, device=device)
            Wk  = nn.Linear(self.d_model, self.d_model, device=device)
            Wv  = nn.Linear(self.d_model, self.d_model, device=device)
            Wo  = nn.Linear(self.d_model, self.d_model, device=device)
            Wcq = nn.Linear(self.d_model, self.d_model, device=device)
            Wck = nn.Linear(self.d_model, self.d_model, device=device)
            Wcv = nn.Linear(self.d_model, self.d_model, device=device)
            Wco = nn.Linear(self.d_model, self.d_model, device=device)

            nn.init.normal_(Wq.weight,  mean=0.0, std=std_qk)
            nn.init.normal_(Wk.weight,  mean=0.0, std=std_qk)
            nn.init.normal_(Wcq.weight, mean=0.0, std=std_qk)
            nn.init.normal_(Wck.weight, mean=0.0, std=std_qk)

            nn.init.normal_(Wv.weight,  mean=0.0, std=std_general)
            nn.init.normal_(Wcv.weight, mean=0.0, std=std_general)

            nn.init.normal_(Wo.weight,  mean=0.0, std=std_general * res_scale)
            nn.init.normal_(Wco.weight, mean=0.0, std=std_general * res_scale)

            for m in [Wq, Wk, Wv, Wo, Wcq, Wck, Wcv, Wco]:
                nn.init.zeros_(m.bias)

            # ── Feed Forward ───────────────────────────────────
            Wff1 = nn.Linear(self.d_model, self.d_ff, device=device)
            Wff2 = nn.Linear(self.d_ff, self.d_model, device=device)

            nn.init.kaiming_normal_(Wff1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(Wff2.weight, mode='fan_in', nonlinearity='relu')
            with torch.no_grad():
                Wff2.weight *= res_scale

            nn.init.zeros_(Wff1.bias)
            nn.init.zeros_(Wff2.bias)

            # ── Append to ModuleLists ──────────────────────────
            W_q.append(Wq);       W_k.append(Wk)
            W_v.append(Wv);       W_o.append(Wo)
            W_cross_q.append(Wcq); W_cross_k.append(Wck)
            W_cross_v.append(Wcv); W_cross_o.append(Wco)
            W_ff1.append(Wff1);   W_ff2.append(Wff2)

            # ── Layer Norms ────────────────────────────────────
            W_norm_self_a.append(nn.Parameter(torch.ones(self.d_model,  device=device)))
            W_norm_self_b.append(nn.Parameter(torch.zeros(self.d_model, device=device)))
            W_norm_cross_a.append(nn.Parameter(torch.ones(self.d_model,  device=device)))
            W_norm_cross_b.append(nn.Parameter(torch.zeros(self.d_model, device=device)))
            W_norm_ff_a.append(nn.Parameter(torch.ones(self.d_model,  device=device)))
            W_norm_ff_b.append(nn.Parameter(torch.zeros(self.d_model, device=device)))

        return (W_q, W_k, W_v, W_o, W_ff1, W_ff2,
                W_norm_self_a, W_norm_self_b,
                W_norm_cross_a, W_norm_cross_b,
                W_norm_ff_a, W_norm_ff_b,
                W_cross_q, W_cross_k, W_cross_v, W_cross_o)

    # ─────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────
 
    def fit_pre(self, X, E,dec_mask_comb,dec_mask_cross_comb):
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

        self.pad_mask_combined = dec_mask_comb

        self.pad_mask_cross= dec_mask_cross_comb
        embeddings = self.emb.weight[X]
        self.D = X
        inputs = embeddings+ self.pos.weight[:len(X[0])]
        inputs= self.dropout['emb'].train(self.is_training).forward(inputs)
        # inputs = embeddings+ self.sin_pos[:len(X[0])]
        # inputs = embeddings
        E= self.dropout['enc_out'].train(self.is_training).forward(E)
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
                self.W_norm_self_a[layer], self.W_norm_self_b[layer],self.epsilon
            )

            multi_self_att, multi_self_att_o, multi_self_att_Q, multi_self_att_K, \
            multi_self_att_V, multi_self_att_A, multi_self_att_raw_A = self.masked_self_att(norm_inputs, layer)

            self_out = inputs +self.dropout['self_att_out'].train(self.is_training).forward( multi_self_att)

            # =========================
            # 🔹 CROSS ATTENTION (Pre-LN)
            # =========================
            norm_self_out = self.layer_norm(
                self_out,
                self.W_norm_cross_a[layer], self.W_norm_cross_b[layer],self.epsilon
            )

            cross_self_att, cross_att_o, cross_att_Q, cross_att_K, cross_att_V, \
             cross_att_A,cross_att_raw_A= self.cross_attention(norm_self_out, E, layer)

            cross_out = self_out + self.dropout['cross_att_out'].train(self.is_training).forward(cross_self_att)

            # =========================
            # 🔹 FEED FORWARD (Pre-LN)
            # =========================
            norm_cross_out = self.layer_norm(
                cross_out,
                self.W_norm_ff_a[layer], self.W_norm_ff_b[layer],self.epsilon
            )

            ff_layer_1 = F.relu( self.W_ff1[layer](norm_cross_out))
            ff_layer_1= self.dropout['ff_layer_1'].train(self.is_training).forward(ff_layer_1)
            ff_output =  self.W_ff2[layer](ff_layer_1)

            

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
            # self.L.append(L)
            # total = self.self_att_a[layer].numel()
            # count = (self.self_att_a[layer] > 1).sum().item()

            # print("count:", count)
            # print("percentage:", count / total)
            inputs = cross_out + self.dropout['ff_out'].train(self.is_training).forward(ff_output)
        self.H = inputs
        # print(self.W_voc.bias.shape)
        logits = inputs @ self.emb.weight.T
        logits = logits
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
        return logits


    def back_pre(self, targets, logits, smoothing=0.1,max_norm=1):


        # rescaled_targets = np.expand_dims(targets, axis=-1)
        # targets_t = torch.tensor(rescaled_targets, dtype=torch.long, device=device)
        # # print(targets_t.shape)
        # mask = (targets_t != 0)
        log_prob = F.log_softmax(logits, dim=-1)   # your current approach (less stable)

        targets_t = targets.unsqueeze(-1)
        mask = (targets_t != 0)

        nll_loss    = -log_prob.gather(dim=-1, index=targets_t)       # (B, T, 1)
        smooth_loss = -log_prob.mean(dim=-1, keepdim=True)             # (B, T, 1)
        
        loss_per_token = (1 - smoothing) * nll_loss + smoothing * smooth_loss
        loss = (loss_per_token * mask).sum() / mask.sum()
        # print(loss)
        delta_z = self.gradient_softmax_cross_entropy(targets, logits, smoothing)

        # apply mask and normalise — same normalisation as loss
        delta_z = delta_z * mask / mask.sum()
        delta_W_voc = self.gradient_W_voc(delta_z).T
        # print(delta_z.shape)
        # delta_W_voc_b= delta_z.sum(dim=(0,1))
        # delta_H = delta_z @ self.emb.weight
        self.delta_H.zero_()
        self.delta_H.copy_(delta_z@ self.emb.weight)
        
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
        

        prev_delta = self.delta_H
        # print(prev_delta.abs().mean())

        del_E = self._del_E
        del_E.zero_()
        layer_grad_store = []
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
                self.W_cross_o[layer].weight.T,
                self.cross_att_o[layer],
                self.cross_att_a[layer],
                self.cross_att_raw_a[layer],
                self.cross_att_v[layer],
                self.cross_att_q[layer],
                self.cross_att_k[layer],
                self.cross_att_inputs[layer],
                self.E,
                self.W_cross_q[layer].weight.T,
                self.W_cross_k[layer].weight.T,
                self.W_cross_v[layer].weight.T
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
                self.W_o[layer].weight.T,
                self.self_att_o[layer],
                self.self_att_a[layer],
                self.self_att_raw_a[layer],
                self.self_att_v[layer],
                self.self_att_q[layer],
                self.self_att_k[layer],
                self.self_att_inputs[layer],
                self.W_q[layer].weight.T,
                self.W_k[layer].weight.T,
                self.W_v[layer].weight.T
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
            layer_grad_store.append({
            'layer': layer,
            # FF
            'ff_w1': ff_w1_grad,    'ff_b1': ff_b1_grad,
            'ff_w2': ff_w2_grad,    'ff_b2': ff_b2_grad,
            # FF norm
            'ff_alpha': ff_alpha_grad, 'ff_beta': ff_beta_grad,
            # Cross attention
            'cq': delta_W_q,  'ck': delta_W_k,  'cv': delta_W_v,  'co': delta_W_o,
            'cq_b': delta_W_q_b, 'ck_b': delta_W_k_b,
            'cv_b': delta_W_v_b, 'co_b': delta_W_o_b,
            # Cross norm
            'cross_alpha': cross_alpha_grad, 'cross_beta': cross_beta_grad,
            # Self attention
            'sq': self_delta_W_q, 'sk': self_delta_W_k,
            'sv': self_delta_W_v, 'so': self_delta_W_o,
            'sq_b': self_delta_W_q_b, 'sk_b': self_delta_W_k_b,
            'sv_b': self_delta_W_v_b, 'so_b': self_delta_W_o_b,
            # Self norm
            'self_alpha': self_alpha_grad, 'self_beta': self_beta_grad,
        })
            # with torch.no_grad():
            #     # FF
            #     self.W_ff1[layer].weight -= self.W_ff1_ada[layer].grad(ff_w1_grad,self.W_ff1[layer].weight)
            #     self.W_ff2[layer].weight -= self.W_ff2_ada[layer].grad(ff_w2_grad,self.W_ff2[layer].weight)
            #     # print((self.W_ff1[layer].weight**2).mean().sqrt(),self.W_ff1_ada[layer].grad(ff_w1_grad,self.W_ff1[layer].weight).abs().mean())
            #     # print(self.W_ff2[layer].weight.abs().mean(),self.W_ff2_ada[layer].grad(ff_w2_grad,self.W_ff1[layer].weight).abs().mean())
            #     self.W_ff1[layer].bias -= self.W_ff1_ada[layer].grad(ff_b1_grad,self.W_ff1[layer].bias)
            #     self.W_ff2[layer].bias-= self.W_ff2_ada[layer].grad(ff_b2_grad,self.W_ff2[layer].bias)
            #     # print(self.b_ff1[layer].abs().mean(),self.b_ff1_ada[layer].grad(ff_b1_grad).abs().mean())
            #     # print(self.b_ff2[layer].abs().mean(),self.b_ff2_ada[layer].grad(ff_b2_grad).abs().mean())
            #     # print(self.W_ff1[layer].weight.abs().mean(),self.W_ff1_ada[layer].grad(ff_w1_grad).abs().mean())

            #     # # Cross
            #     self.W_cross_q[layer].weight -= self.W_cross_q_ada[layer].grad(delta_W_q,self.W_cross_q[layer].weight)
            #     self.W_cross_k[layer].weight -= self.W_cross_k_ada[layer].grad(delta_W_k,self.W_cross_k[layer].weight)
            #     self.W_cross_v[layer].weight -= self.W_cross_v_ada[layer].grad(delta_W_v,self.W_cross_v[layer].weight)
            #     self.W_cross_o[layer].weight -= self.W_cross_o_ada[layer].grad(delta_W_o,self.W_cross_o[layer].weight)
            #     self.W_cross_q[layer].bias -= self.W_cross_q_ada[layer].grad(delta_W_q_b,self.W_cross_q[layer].bias)
            #     self.W_cross_k[layer].bias -= self.W_cross_k_ada[layer].grad(delta_W_k_b,self.W_cross_k[layer].bias)
            #     self.W_cross_v[layer].bias -= self.W_cross_v_ada[layer].grad(delta_W_v_b,self.W_cross_v[layer].bias)
            #     self.W_cross_o[layer].bias -= self.W_cross_o_ada[layer].grad(delta_W_o_b,self.W_cross_o[layer].bias)
            #     # print(self.W_cross_q[layer].weight.abs().mean(),self.W_cross_q_ada[layer].grad(delta_W_q,self.W_cross_q[layer].weight).abs().mean())
            #     # print(self.W_cross_k[layer].weight.abs().mean(),self.W_cross_k_ada[layer].grad(delta_W_k).abs().mean())
            #     # print(self.W_cross_v[layer].weight.abs().mean(),self.W_cross_v_ada[layer].grad(delta_W_v).abs().mean())
            #     # print(self.W_cross_o[layer].weight.abs().mean(),self.W_cross_o_ada[layer].grad(delta_W_o).abs().mean())
            #     # # Self
            #     self.W_q[layer].weight -= self.W_q_ada[layer].grad(self_delta_W_q,self.W_q[layer].weight)
            #     self.W_k[layer].weight -= self.W_k_ada[layer].grad(self_delta_W_k,self.W_k[layer].weight)
            #     self.W_v[layer].weight -= self.W_v_ada[layer].grad(self_delta_W_v,self.W_v[layer].weight)
            #     self.W_o[layer].weight -= self.W_o_ada[layer].grad(self_delta_W_o,self.W_o[layer].weight)
            #     self.W_q[layer].bias -= self.W_q_ada[layer].grad(self_delta_W_q_b,self.W_q[layer].bias)
            #     self.W_k[layer].bias -= self.W_k_ada[layer].grad(self_delta_W_k_b,self.W_k[layer].bias)
            #     self.W_v[layer].bias -= self.W_v_ada[layer].grad(self_delta_W_v_b,self.W_v[layer].bias)
            #     self.W_o[layer].bias -= self.W_o_ada[layer].grad(self_delta_W_o_b,self.W_o[layer].bias)
            #     # print(self.W_q[layer].weight.abs().mean(),self.W_q_ada[layer].grad(self_delta_W_q).abs().mean())
            #     # print(self.W_k[layer].weight.abs().mean(),self.W_k_ada[layer].grad(self_delta_W_k).abs().mean())
            #     # print(self.W_v[layer].weight.abs().mean(),self.W_v_ada[layer].grad(self_delta_W_v).abs().mean())
            #     # print(self.W_o[layer].weight.abs().mean(),self.W_o_ada[layer].grad(self_delta_W_o).abs().mean())
            #     # # # Norms
            #     self.W_norm_self_a[layer] -= self.W_norm_self_a_ada[layer].grad(self_alpha_grad,self.W_norm_self_a[layer])
            #     self.W_norm_self_b[layer] -= self.W_norm_self_b_ada[layer].grad(self_beta_grad,self.W_norm_self_b[layer])
            #     # print(abs_mean(self.W_norm_self_a[layer]),abs_mean(self.W_norm_self_a_ada[layer].grad(self_alpha_grad)))
            #     # print(abs_mean(self.W_norm_self_b[layer]),abs_mean(self.W_norm_self_b_ada[layer].grad(self_beta_grad)))
            #     self.W_norm_cross_a[layer] -= self.W_norm_cross_a_ada[layer].grad(cross_alpha_grad,self.W_norm_cross_a[layer])
            #     self.W_norm_cross_b[layer] -= self.W_norm_cross_b_ada[layer].grad(cross_beta_grad,self.W_norm_cross_b[layer])
            #     # print(abs_mean(self.W_norm_cross_a[layer]),abs_mean(self.W_norm_cross_a_ada[layer].grad(cross_alpha_grad)))
            #     # print(abs_mean(self.W_norm_cross_b[layer]),abs_mean(self.W_norm_cross_b_ada[layer].grad(cross_beta_grad)))
            #     self.W_norm_ff_a[layer] -= self.W_norm_ff_a_ada[layer].grad(ff_alpha_grad,self.W_norm_ff_a[layer])
            #     self.W_norm_ff_b[layer] -= self.W_norm_ff_b_ada[layer].grad(ff_beta_grad,self.W_norm_ff_b[layer] )
            #     # print(abs_mean(self.W_norm_ff_a[layer] ),abs_mean(self.W_norm_ff_a_ada[layer].grad(ff_alpha_grad)))
            #     # print(abs_mean(self.W_norm_ff_b[layer] ),abs_mean(self.W_norm_ff_b_ada[layer].grad(ff_beta_grad)))
            #     pass
        # print(prev_delta)
        del_E= self.dropout['enc_out'].train(self.is_training).backward(del_E)
        prev_delta  = self.dropout['emb'].backward(prev_delta)
        flat_tokens = self.D.view(-1)
        flat_delta  = prev_delta.view(-1, self.d_model)
        
        self.emb_grad.zero_()
        self.emb_grad.index_add_(0, flat_tokens, flat_delta)
        delta_W_voc += self.emb_grad
        
        all_grads = [delta_W_voc]
        for store in layer_grad_store:
            all_grads += [
                store['ff_w1'],  store['ff_b1'],  store['ff_w2'],  store['ff_b2'],
                store['ff_alpha'], store['ff_beta'],
                store['cq'],  store['ck'],  store['cv'],  store['co'],
                store['cq_b'], store['ck_b'], store['cv_b'], store['co_b'],
                store['cross_alpha'], store['cross_beta'],
                store['sq'],  store['sk'],  store['sv'],  store['so'],
                store['sq_b'], store['sk_b'], store['sv_b'], store['so_b'],
                store['self_alpha'], store['self_beta'],
            ]

        # coef = self.clip_grad_norm(all_grads, max_norm)
        self.grads=layer_grad_store
        # for store in layer_grad_store:
        #     layer = store['layer']
        #     with torch.no_grad():
        #         # FF
        #         self.W_ff1[layer].weight -= self.W_ff1_ada[layer].grad(store['ff_w1'] * coef, self.W_ff1[layer].weight)
        #         self.W_ff1[layer].bias   -= self.W_ff1_ada[layer].grad(store['ff_b1'] * coef, self.W_ff1[layer].bias)
        #         self.W_ff2[layer].weight -= self.W_ff2_ada[layer].grad(store['ff_w2'] * coef, self.W_ff2[layer].weight)
        #         self.W_ff2[layer].bias   -= self.W_ff2_ada[layer].grad(store['ff_b2'] * coef, self.W_ff2[layer].bias)
        #         # FF norm
        #         self.W_norm_ff_a[layer]  -= self.W_norm_ff_a_ada[layer].grad(store['ff_alpha'] * coef, self.W_norm_ff_a[layer])
        #         self.W_norm_ff_b[layer]  -= self.W_norm_ff_b_ada[layer].grad(store['ff_beta']  * coef, self.W_norm_ff_b[layer])
        #         # Cross attention
        #         self.W_cross_q[layer].weight -= self.W_cross_q_ada[layer].grad(store['cq']   * coef, self.W_cross_q[layer].weight)
        #         self.W_cross_k[layer].weight -= self.W_cross_k_ada[layer].grad(store['ck']   * coef, self.W_cross_k[layer].weight)
        #         self.W_cross_v[layer].weight -= self.W_cross_v_ada[layer].grad(store['cv']   * coef, self.W_cross_v[layer].weight)
        #         self.W_cross_o[layer].weight -= self.W_cross_o_ada[layer].grad(store['co']   * coef, self.W_cross_o[layer].weight)
        #         self.W_cross_q[layer].bias   -= self.W_cross_q_ada[layer].grad(store['cq_b'] * coef, self.W_cross_q[layer].bias)
        #         self.W_cross_k[layer].bias   -= self.W_cross_k_ada[layer].grad(store['ck_b'] * coef, self.W_cross_k[layer].bias)
        #         self.W_cross_v[layer].bias   -= self.W_cross_v_ada[layer].grad(store['cv_b'] * coef, self.W_cross_v[layer].bias)
        #         self.W_cross_o[layer].bias   -= self.W_cross_o_ada[layer].grad(store['co_b'] * coef, self.W_cross_o[layer].bias)
        #         # Cross norm
        #         self.W_norm_cross_a[layer] -= self.W_norm_cross_a_ada[layer].grad(store['cross_alpha'] * coef, self.W_norm_cross_a[layer])
        #         self.W_norm_cross_b[layer] -= self.W_norm_cross_b_ada[layer].grad(store['cross_beta']  * coef, self.W_norm_cross_b[layer])
        #         # Self attention
        #         self.W_q[layer].weight -= self.W_q_ada[layer].grad(store['sq']   * coef, self.W_q[layer].weight)
        #         self.W_k[layer].weight -= self.W_k_ada[layer].grad(store['sk']   * coef, self.W_k[layer].weight)
        #         self.W_v[layer].weight -= self.W_v_ada[layer].grad(store['sv']   * coef, self.W_v[layer].weight)
        #         self.W_o[layer].weight -= self.W_o_ada[layer].grad(store['so']   * coef, self.W_o[layer].weight)
        #         self.W_q[layer].bias   -= self.W_q_ada[layer].grad(store['sq_b'] * coef, self.W_q[layer].bias)
        #         self.W_k[layer].bias   -= self.W_k_ada[layer].grad(store['sk_b'] * coef, self.W_k[layer].bias)
        #         self.W_v[layer].bias   -= self.W_v_ada[layer].grad(store['sv_b'] * coef, self.W_v[layer].bias)
        #         self.W_o[layer].bias   -= self.W_o_ada[layer].grad(store['so_b'] * coef, self.W_o[layer].bias)
        #         # Self norm
        #         self.W_norm_self_a[layer] -= self.W_norm_self_a_ada[layer].grad(store['self_alpha'] * coef, self.W_norm_self_a[layer])
        #         self.W_norm_self_b[layer] -= self.W_norm_self_b_ada[layer].grad(store['self_beta']  * coef, self.W_norm_self_b[layer])
        
        return del_E,loss,delta_W_voc,torch.sum(prev_delta, dim=0),all_grads

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
    def gradient_softmax_cross_entropy(self, targets, logits, smoothing=0.1):
        self.delta_z.zero_()
        prob = torch.softmax(logits, dim=-1)
        self.delta_z.copy_(prob)
        self.delta_z[torch.arange(self.batch_size)[:, None], torch.arange(logits.shape[1]), targets] -= (1.0 - smoothing)
        self.delta_z -= smoothing / self.voc_size
        delta_z = self.delta_z
        return delta_z

        # subtract smoothing/V from every vocab position
        # this is the gradient of the uniform distribution penalty term
        # mathematically: d/dz [ -eps * mean(log p) ] = eps/V * (p - 1/p... )
        # simplified: just subtract eps/V from every position since d(-log p_i)/dz_j = p_j - 1_{i==j}
        # summed uniformly: p_j - 1/V  →  already captured by subtracting 1/V                                       # (B, T, V)

    def gradient_W_voc(self, delta_z):
        #  return (self.H.view(-1, self.d_model).T @ delta_z.view(-1, self.voc_size))
         return self._weight_grad(self.H,delta_z)

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
        delta_ff_w2 = self._weight_grad(layer_1_output,delta).T
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
        # delta_ff_w1= (x.view(-1, self.d_model).T @ delta_a.view(-1, self.d_ff)).T
        delta_ff_w1=self._weight_grad(x,delta_a).T
        delta_x = delta_a @ ff_w_1.T
        
        return delta_ff_w1, delta_ff_b1, delta_ff_w2, delta_ff_b2, delta_x

    def grad_cross_att(self, delta, W_o, O, A,raw_A, V, Q, K, X, E, W_q, W_k, W_v):
        B, T, _ = delta.shape
        delta_o = delta @ W_o.T
        # delta_W_o = torch.einsum('bti,btj->ij', O, delta)
        # delta_W_o=(O.view(-1, self.d_model).T @ delta.view(-1, self.d_model))
        delta_W_o =self._weight_grad(O,delta)
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
        heads_delta_V= heads_delta_V.transpose(1, 2).reshape(B, T, -1)   
        heads_delta_Q= heads_delta_Q.transpose(1, 2).reshape(B, T, -1) 
        heads_delta_K= heads_delta_K.transpose(1,2).reshape(B, T, -1)
        delta_E_K = heads_delta_K @ W_k.T
        delta_E_V = heads_delta_V @ W_v.T
        delta_X_Q = heads_delta_Q @ W_q.T
        # delta_W_q=(X.view(-1, self.d_model).T @ heads_delta_Q.view(-1, self.d_model))
        # delta_W_k=(E.view(-1, self.d_model).T @ heads_delta_K.view(-1, self.d_model))
        # delta_W_v=(E.view(-1, self.d_model).T @ heads_delta_V.view(-1, self.d_model))
        delta_W_q=self._weight_grad(X,heads_delta_Q)
        delta_W_k=self._weight_grad(E,heads_delta_K)
        delta_W_v= self._weight_grad(E,heads_delta_V)     
        delta_W_q_b=  heads_delta_Q.sum(dim=(0,1))
        delta_W_k_b = heads_delta_K.sum(dim=(0,1))
        delta_W_v_b=  heads_delta_V.sum(dim=(0,1))
        return delta_E_K, delta_E_V, delta_X_Q, delta_W_q, delta_W_k, delta_W_v, delta_W_o,delta_W_q_b,delta_W_k_b,delta_W_v_b,delta_W_o_b

    def grad_multi_self_att(self, delta, W_o, O, A,raw_A, V, Q, K, X, W_q, W_k, W_v):
        B, T, _ = delta.shape
        delta_o = delta @ W_o.T
        
        # delta_W_o = torch.einsum('bti,btj->ij', O, delta)
        # delta_W_o=(O.view(-1, self.d_model).T @ delta.view(-1, self.d_model))\
        
        delta_W_o = self._weight_grad(O,delta)
        delta_W_o_b= delta.sum(dim=(0,1))
        heads_delta_o = delta_o.view(B, T, self.h_count, self.d_k).transpose(1, 2)
        # print(heads_delta_o)
        # print(heads_delta_o.shape,V.shape)
        heads_delta_A= heads_delta_o@V.transpose(-2,-1)
        heads_delta_A= self.dropout['self_att_a'].backward(heads_delta_A)
        heads_delta_V= A.transpose(-2,-1)@heads_delta_o
        dot_delA_A= (heads_delta_A * raw_A).sum(dim=-1, keepdim=True)
        heads_delta_S = raw_A * (heads_delta_A - dot_delA_A)
        
        heads_delta_Q= (heads_delta_S @ K) / np.sqrt(self.d_k)
        heads_delta_K= (heads_delta_S.transpose(-2, -1) @ Q) / np.sqrt(self.d_k)
        heads_delta_V= heads_delta_V.transpose(1, 2).reshape(B, T, -1)  
        heads_delta_Q= heads_delta_Q.transpose(1, 2).reshape(B, T, -1)  
        heads_delta_K= heads_delta_K.transpose(1,2).reshape(B, T, -1) 
        delta_X_K = heads_delta_K @ W_k.T
        delta_X_V = heads_delta_V @ W_v.T
        delta_X_Q = heads_delta_Q @ W_q.T
        # dq,dk,dv=flash_attention_backward(Q,K,V,O.view(B, T, self.h_count, self.d_k).transpose(1,2),heads_delta_o,L)

        # delta_W_q= torch.einsum('bti,btj->ij', X, heads_delta_Q)
        # delta_W_k = torch.einsum('bti,btj->ij', X, heads_delta_K)
        # delta_W_v = torch.einsum('bti,btj->ij', X, heads_delta_V)
        # delta_W_q=(X.view(-1, self.d_model).T @ heads_delta_Q.view(-1, self.d_model))
        # delta_W_k=(X.view(-1, self.d_model).T @ heads_delta_K.view(-1, self.d_model))
        # delta_W_v=(X.view(-1, self.d_model).T @ heads_delta_V.view(-1, self.d_model))
        delta_W_q=self._weight_grad(X,heads_delta_Q)
        delta_W_k=self._weight_grad(X,heads_delta_K)
        delta_W_v= self._weight_grad(X,heads_delta_V)  
        delta_W_q_b=  heads_delta_Q.sum(dim=(0,1))
        delta_W_k_b = heads_delta_K.sum(dim=(0,1))
        delta_W_v_b=  heads_delta_V.sum(dim=(0,1))
   
        return delta_X_K, delta_X_V, delta_X_Q, delta_W_q, delta_W_k, delta_W_v, delta_W_o,delta_W_q_b,delta_W_k_b,delta_W_v_b,delta_W_o_b

    # ─────────────────────────────────────────────
    # Attention & norm
    # ─────────────────────────────────────────────
    def masked_self_att(self, x, layer):
        B, T, _ = x.shape
        Q = (self.W_q[layer](x)).view(B, T, self.h_count, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = (self.W_k[layer](x)).view(B, T, self.h_count, self.d_k).transpose(1, 2)
        V = (self.W_v[layer](x)).view(B, T, self.h_count, self.d_k).transpose(1, 2)
        temp=1
        S = Q @ K.transpose(-2, -1) /(math.sqrt(self.d_k)*temp)  
        pad_mask = self.pad_mask_combined    # (B, H, T, T)
        S = S + self.causal_mask[:T, :T]
        S= S.masked_fill(pad_mask,negative_inf)
        # S=S - S.max(dim=-1, keepdim=True).values
        A = F.softmax(S, dim=-1)  
        A = torch.nan_to_num(A, nan=0.0)
        raw_A= A
        A= self.dropout['self_att_a'].train(self.is_training).forward(A)                                  # (B, H, T, T)
        O = (A @ V).transpose(1, 2).reshape(B, T, -1)   # (B, T, d_model)
        # O_flash,L=flash_attention_pytorch(Q,K,V,True,None,self.dec_k,self.dec_q)
        # print(O[0][0]==O_flash.transpose(1, 2).reshape(B, T, -1)[0][0])
        output = self.W_o[layer](O)
        return output, O, Q, K, V, A,raw_A

    def cross_attention(self, z1, E, layer):
        B, T, _ = z1.shape
        E_B,E_T,_= E.shape
        Wq = self.W_cross_q[layer]
        Wk = self.W_cross_k[layer]
        Wv = self.W_cross_v[layer]
        # Wq_b = self.W_cross_q[layer].bias
        # Wk_b = self.W_cross_k[layer].bias
        # Wv_b = self.W_cross_v[layer].bias
        # print(Wk.shape)
        Q = ( Wq(z1)).view(B, T, self.h_count, self.d_k).transpose(1, 2)
        K = (Wk(E)).view(E_B, E_T, self.h_count, self.d_k).transpose(1, 2)
        V = (Wv(E)).view(E_B, E_T, self.h_count, self.d_k).transpose(1, 2)
        temp=1
        S = Q @ K.transpose(-2, -1) / (math.sqrt(self.d_k)*temp)        # (B, H, T, T)
        # S = S - S.max(dim=-1, keepdim=True)[0]
        pad_mask = self.pad_mask_cross 
        S= S.masked_fill(pad_mask,negative_inf)
        A = F.softmax(S, dim=-1)  
        A = torch.nan_to_num(A, nan=0.0)
        
        raw_A= A   
        A= self.dropout['cross_att_a'].train(self.is_training).forward(A)                               # (B, H, T, T)
        O = (A @ V).transpose(1, 2).reshape(B, T, -1)
        output = self.W_cross_o[layer](O)

        return output, O, Q, K, V,  A,raw_A

    def layer_norm(self, X, alpha, beta,epsilon):
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + epsilon)
        return ((X - mean) / std) * alpha + beta
    
    def update_weights(self,coef):
         for store in self.grads:
            layer = store['layer']
            with torch.no_grad():
                # FF
                self.W_ff1[layer].weight -= self.W_ff1_ada[layer].grad(store['ff_w1'] * coef, self.W_ff1[layer].weight)
                self.W_ff1[layer].bias   -= self.W_ff1_ada[layer].grad(store['ff_b1'] * coef, self.W_ff1[layer].bias)
                self.W_ff2[layer].weight -= self.W_ff2_ada[layer].grad(store['ff_w2'] * coef, self.W_ff2[layer].weight)
                self.W_ff2[layer].bias   -= self.W_ff2_ada[layer].grad(store['ff_b2'] * coef, self.W_ff2[layer].bias)
                # FF norm
                self.W_norm_ff_a[layer]  -= self.W_norm_ff_a_ada[layer].grad(store['ff_alpha'] * coef, self.W_norm_ff_a[layer])
                self.W_norm_ff_b[layer]  -= self.W_norm_ff_b_ada[layer].grad(store['ff_beta']  * coef, self.W_norm_ff_b[layer])
                # Cross attention
                self.W_cross_q[layer].weight -= self.W_cross_q_ada[layer].grad(store['cq'].T   * coef, self.W_cross_q[layer].weight)
                self.W_cross_k[layer].weight -= self.W_cross_k_ada[layer].grad(store['ck'].T   * coef, self.W_cross_k[layer].weight)
                self.W_cross_v[layer].weight -= self.W_cross_v_ada[layer].grad(store['cv'].T   * coef, self.W_cross_v[layer].weight)
                self.W_cross_o[layer].weight -= self.W_cross_o_ada[layer].grad(store['co'].T  * coef, self.W_cross_o[layer].weight)
                self.W_cross_q[layer].bias   -= self.W_cross_q_ada[layer].grad(store['cq_b'] * coef, self.W_cross_q[layer].bias)
                self.W_cross_k[layer].bias   -= self.W_cross_k_ada[layer].grad(store['ck_b'] * coef, self.W_cross_k[layer].bias)
                self.W_cross_v[layer].bias   -= self.W_cross_v_ada[layer].grad(store['cv_b'] * coef, self.W_cross_v[layer].bias)
                self.W_cross_o[layer].bias   -= self.W_cross_o_ada[layer].grad(store['co_b'] * coef, self.W_cross_o[layer].bias)
                # Cross norm
                self.W_norm_cross_a[layer] -= self.W_norm_cross_a_ada[layer].grad(store['cross_alpha'] * coef, self.W_norm_cross_a[layer])
                self.W_norm_cross_b[layer] -= self.W_norm_cross_b_ada[layer].grad(store['cross_beta']  * coef, self.W_norm_cross_b[layer])
                # Self attention
                self.W_q[layer].weight -= self.W_q_ada[layer].grad(store['sq'].T   * coef, self.W_q[layer].weight)
                self.W_k[layer].weight -= self.W_k_ada[layer].grad(store['sk'].T   * coef, self.W_k[layer].weight)
                self.W_v[layer].weight -= self.W_v_ada[layer].grad(store['sv'].T   * coef, self.W_v[layer].weight)
                self.W_o[layer].weight -= self.W_o_ada[layer].grad(store['so'].T   * coef, self.W_o[layer].weight)
                self.W_q[layer].bias   -= self.W_q_ada[layer].grad(store['sq_b'] * coef, self.W_q[layer].bias)
                self.W_k[layer].bias   -= self.W_k_ada[layer].grad(store['sk_b'] * coef, self.W_k[layer].bias)
                self.W_v[layer].bias   -= self.W_v_ada[layer].grad(store['sv_b'] * coef, self.W_v[layer].bias)
                self.W_o[layer].bias   -= self.W_o_ada[layer].grad(store['so_b'] * coef, self.W_o[layer].bias)
                # Self norm
                self.W_norm_self_a[layer] -= self.W_norm_self_a_ada[layer].grad(store['self_alpha'] * coef, self.W_norm_self_a[layer])
                self.W_norm_self_b[layer] -= self.W_norm_self_b_ada[layer].grad(store['self_beta']  * coef, self.W_norm_self_b[layer])
        
    def clip_grad_norm(self, all_grads, max_norm=1.0):
        """Compute global norm across all encoder gradients and return clip coefficient."""
        total_norm = 0.0
        for g in all_grads:
            if g is not None:
                total_norm += g.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
        # print(f"[Encoder ClipGrad] global_norm={total_norm:.4f}, coef={clip_coef:.4f}")
        return clip_coef
    def _weight_grad(self, x, delta):
    # x: (B,T,Di), delta: (B,T,Do) → (Di, Do)
       return torch.einsum('bti,btj->ij', x, delta)
    
    def create_pad_mask_q(self, X):

            pad_q = (X == 0).unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
            # pad_k = (X == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            return pad_q  
    def create_pad_mask_k(self, X):

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
        self.grads.clear()
        
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

def _mask_k(x: torch.Tensor) -> torch.Tensor:
    """Key-side pad mask  (B, 1, 1, T)  –  True where token == 0."""
    return (x == 0).unsqueeze(1).unsqueeze(2)


def _mask_q(x: torch.Tensor) -> torch.Tensor:
    """Query-side pad mask  (B, 1, T, 1)  –  True where token == 0."""
    return (x == 0).unsqueeze(1).unsqueeze(3)

def predict(y, target, encoder, decoder, max_len=128):
    encoder.val()
    decoder.val()
    enc_pad_k= _mask_k(y)
    enc_pad_q= _mask_q(y)
    E = encoder.fit_pre(y,enc_pad_k|enc_pad_q)

    # full padded sequence
    start = torch.zeros((1, max_len), dtype=torch.long, device=device)

    # BOS token
    start[0, 0] = 2
    
    print(tokenizer.decode(y[0].tolist()))

    i = 0
    while i < max_len - 1 and start[0,len(start[0])-1]!=3:
        pad_mask_k= _mask_k(start)
        pad_mask_q= _mask_q(start)
        logits = decoder.fit_pre(start, E,pad_mask_k|pad_mask_q,enc_pad_k|pad_mask_q) 
        prob= F.softmax(logits,dim=-1)
        
        token_prob, index = torch.max(prob, dim=-1)
         
        next_token = index[0, i].item()
        print(tokenizer.decode([next_token]), end=" ")

        # insert next token into padded tensor
        start[0, i + 1] = next_token

        # EOS token
        if next_token == 3:
            break

        i += 1
    encoder.clear_memory()
    decoder.clear_memory()
def calculate_validation_loss(encoder, decoder, x_val_encoder, x_val_decoder, x_val_target,enc_k,enc_q,dec_k,dec_q, batch_size=64):
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
            E    = encoder.fit_pre(enc_batch,enc_k[i : i + batch_size]|enc_q[i : i + batch_size])
            logits = decoder.fit_pre(dec_batch, E,dec_k[i : i + batch_size]|dec_q[i : i + batch_size],enc_k[i : i + batch_size]|dec_q[i : i + batch_size])

            # ---------- loss (mirrors your back_pre logic) ----------
            targets_t = tgt_batch.unsqueeze(-1)
            # print(targets_t.shape)
            log_prob = F.log_softmax(logits, dim=-1)   # your current approach (less stable)

            mask = (targets_t != 0)

            nll_loss    = -log_prob.gather(dim=-1, index=targets_t)       # (B, T, 1)
            # smooth_loss = -log_prob.mean(dim=-1, keepdim=True)             # (B, T, 1)
            
            loss_per_token =  nll_loss 
            loss = (loss_per_token * mask).sum() / mask.sum()

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
