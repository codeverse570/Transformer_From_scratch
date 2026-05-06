import math

import torch
import torch.nn as nn

import itertools
import json
import torch.nn.functional as F
import numpy as np
from dropout import Dropout
from adam import AdamCustom
# from utils.utils import create_training_files,create_tokens_from_file,get_tokens_from_file,make_samples,abs_mean
# from AdamCustom import AdamCustom
from pathlib import Path
negative_inf = float('-inf')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

class Encoder():
    def __init__(self,d_model,h_count,d_ff,voc_size,sent_len,layers,batch_size,schedular):
      self.positional_encoding=[]
      self.encodings=[]
      self.d_model=d_model
      self.h_count=h_count
      self.d_ff=d_ff
      self.d_k= d_model//h_count
    #   self.dropout= Dropout()
      self.voc_size= voc_size
      self.sent_len= sent_len
      self.pad_mask=[]
      self.dropout= {'emb':Dropout(0),'self_att_a':Dropout(),'self_att_out':Dropout(),'ff_layer_1':Dropout(),'ff_out':Dropout()}
      self.schedular=schedular
      self.emb = nn.Embedding(voc_size,self.d_model,device=device)
      self.pos= nn.Embedding(sent_len,self.d_model,device=device)
      self.sin_pos= self.get_sinusoidal_positional_encoding(self.sent_len,self.d_model)
      self.emb_ad= AdamCustom(0.99,self.voc_size,self.d_model,0.01,schedular=self.schedular)
      self.pos_ada= AdamCustom(0.99,self.sent_len,self.d_model,0.01,schedular=self.schedular)
      self.layers=layers
      self.is_training=True
      self.batch_size=batch_size
      self.W_q,self.W_k,self.W_v,self.W_o,self.W_ff1,self.W_ff2,self.W_norm_ff_a,self.W_norm_ff_b,self.W_norm_att_a,self.W_norm_att_b= self.intialize_weights()
      self.W_q_ada,self.W_k_ada,self.W_v_ada,self.W_o_ada,\
      self.W_ff1_ada,self.W_ff2_ada,\
      self.W_norm_self_a_ada,self.W_norm_self_b_ada,\
      self.W_norm_ff_a_ada,self.W_norm_ff_b_ada = self.intialize_optimizers()
      self.b_ff1= [ nn.Parameter(torch.zeros(self.d_ff,device=device)) for i in range(self.layers)]
      self.b_ff2 = [ nn.Parameter(torch.zeros(self.d_model,device=device)) for i in range(self.layers)]
      self.b_ff1_ada= [ AdamCustom(0.99,self.d_ff,self.d_ff,0.01,schedular=self.schedular) for i in range(self.layers)]
      self.b_ff2_ada = [ AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular) for i in range(self.layers)]
      self.ff_output=[]
      self.ff_inputs=[]
      self.ff_layer_1_output=[]
      self.att_output=[]
      self.att_inputs=[]
      self.att_o=[]
      self.att_q=[]
      self.att_k=[]
      self.att_v=[]
      self.att_s=[]
      self.att_a=[] 
      self.att_raw_a=[]       
      self.grads=[]
      self.epsilon = 0.001
    #   nn.init.xavier_normal_(self.emb.weight)
    #   nn.init.xavier_normal_(self.pos.weight)

    def intialize_weights(self):
       W_q=[]
       W_k=[]
       W_v=[]
       W_o=[]
       W_ff1=[]
       W_ff2=[]
       W_norm_ff_a=[]
       W_norm_ff_b=[]
       W_norm_att_a=[]
       W_norm_att_b=[]
       for i in range(self.layers):
          layer_mat_W_q = nn.Linear(self.d_model, self.d_model).to(device)
          layer_mat_W_k = nn.Linear(self.d_model, self.d_model).to(device)
          layer_mat_W_v = nn.Linear(self.d_model, self.d_model).to(device)
          layer_mat_W_o = nn.Linear(self.d_model, self.d_model).to(device)

          layer_ff_W1 = nn.Linear(self.d_model, self.d_ff).to(device)
          layer_ff_W2 = nn.Linear(self.d_ff, self.d_model).to(device)

          layer_ff_norm_a = nn.Parameter(torch.ones(self.d_model, device=device))
          layer_ff_norm_b = nn.Parameter(torch.zeros(self.d_model, device=device))

          layer_att_norm_a = nn.Parameter(torch.ones(self.d_model, device=device))
          layer_att_norm_b = nn.Parameter(torch.zeros(self.d_model, device=device))
          for m in [layer_ff_W1,layer_ff_W2,layer_mat_W_k,layer_mat_W_q,layer_mat_W_v,layer_mat_W_o]:
                 nn.init.xavier_uniform_(m.weight)
                 nn.init.zeros_(m.bias)
          W_q.append(layer_mat_W_q)
          W_k.append(layer_mat_W_k)
          W_v.append(layer_mat_W_v)
          W_o.append(layer_mat_W_o)
          W_ff1.append(layer_ff_W1)
          W_ff2.append(layer_ff_W2)
          W_norm_ff_a.append(layer_ff_norm_a)
          W_norm_ff_b.append(layer_ff_norm_b)
          W_norm_att_a.append(layer_att_norm_a)
          W_norm_att_b.append(layer_att_norm_b)
       return W_q,W_k,W_v,W_o,W_ff1,W_ff2,W_norm_ff_a,W_norm_ff_b,W_norm_att_a,W_norm_att_b

    def intialize_optimizers(self):
       W_q_ada=[]
       W_k_ada=[]
       W_v_ada=[]
       W_o_ada=[] 
       W_ff1_ada=[]
       W_ff2_ada=[] 
       W_norm_self_a_ada=[]
       W_norm_self_b_ada=[]   
       W_norm_ff_a_ada=[]
       W_norm_ff_b_ada=[]      
       for i in range(self.layers):
          W_q_ada.append(AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular))
          W_k_ada.append(AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular))
          W_v_ada.append(AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular))
          W_o_ada.append(AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular))
          W_ff1_ada.append(AdamCustom(0.99,self.d_ff,self.d_model,0.01,schedular=self.schedular))
          W_ff2_ada.append(AdamCustom(0.99,self.d_model,self.d_ff,0.01,schedular=self.schedular))
          W_norm_self_a_ada.append(AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular))
          W_norm_self_b_ada.append(AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular))  
          W_norm_ff_a_ada.append(AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular))
          W_norm_ff_b_ada.append(AdamCustom(0.99,self.d_model,self.d_model,0.01,schedular=self.schedular))
       return W_q_ada,W_k_ada,W_v_ada,W_o_ada,W_ff1_ada,W_ff2_ada,W_norm_self_a_ada,W_norm_self_b_ada,W_norm_ff_a_ada,W_norm_ff_b_ada

    def forward(self,x):
       pass
    

    def fit_pre(self, x):
     
        self.clear_memory()
        self.pad_mask_k=self.create_pad_mask_k(x)
        self.pad_mask_q= self.create_pad_mask_q(x)
        x_encodings = self.emb.weight[x]
        x_encodings = x_encodings + (self.pos.weight)
        # x_encodings = x_encodings*torch.sqrt(self.d_model) + self.sin_pos* 0.02

        # pos_encoding= self.get_sinusoidal_positional_encoding(len(x[0]),self.d_model)

        inputs = self.dropout['emb'].train(self.is_training).forward(x_encodings)
        # inputs = x_encodings
        # store residuals (needed if you later fix backward fully)
        self.residual_att = []
        self.residual_ff = []

        for layer in range(self.layers):
        
            # =========================
            # 🔹 SELF ATTENTION (Pre-LN)
            # =========================
            norm_inputs = self.layer_norm(
                inputs,
                self.W_norm_att_a[layer],
                self.W_norm_att_b[layer]
            )

            att_output, att_o, att_Q, att_K, att_V, att_S, att_A,raw_A= self.self_attention(norm_inputs, layer)

            att_residual = inputs + self.dropout['self_att_out'].train(self.is_training).forward(att_output)

            # =========================
            # 🔹 FEED FORWARD (Pre-LN)
            # =========================
            norm_att = self.layer_norm(
                att_residual,
                self.W_norm_ff_a[layer],
                self.W_norm_ff_b[layer]
            )

            ff_layer_1 = F.relu(norm_att @ self.W_ff1[layer].weight.T +self.W_ff1[layer].bias )
            ff_layer_1= self.dropout['ff_layer_1'].train(self.is_training).forward(ff_layer_1)
            ff_output = ff_layer_1 @ self.W_ff2[layer].weight.T + self.W_ff2[layer].bias

            outputs = att_residual + self.dropout['ff_out'].train(self.is_training).forward(ff_output)

            # =========================
            # 🔹 STORE ACTIVATIONS
            # =========================

            # residuals
            self.residual_att.append(inputs )
            self.residual_ff.append(att_residual )

            # FF
            self.ff_output.append(ff_output )
            self.ff_inputs.append(norm_att )   # PRE-LN input
            self.ff_layer_1_output.append(ff_layer_1 )

            # Attention
            self.att_output.append(att_output )
            self.att_inputs.append(norm_inputs )  # PRE-LN input
            self.att_o.append(att_o )
            self.att_q.append(att_Q )
            self.att_k.append(att_K )
            self.att_v.append(att_V )
            self.att_s.append(att_S )
            self.att_a.append(att_A )
            self.att_raw_a.append(raw_A)
            # total = self.att_a[layer].numel()
            # count = (self.att_a[layer] > 1).sum().item()

            # print("count:", count)
            # print("percentage:", count / total)
            inputs = outputs

        return inputs,self.pad_mask_k


    def back_pre(self,delta,E,tags,max_norm=1):
         new_delta=delta
         layer_grad_store = []
        #  print(new_delta.abs().mean())
         for layer in range(self.layers-1,-1,-1):
            #  print(f'layer-{layer} | {new_delta.abs().mean()}')
             ff_w1_grad,ff_b1_grad,ff_w2_grad,ff_b2_grad,ff_x_grad=self.grad_ff(self.dropout['ff_out'].backward(new_delta),self.ff_layer_1_output[layer],self.W_ff2[layer].weight.T,self.W_ff1[layer].weight.T,self.ff_inputs[layer])
             del_ff_r,del_ff_alpha,del_ff_beta= self.grad_layer_norm_pre(ff_x_grad,self.W_norm_ff_a[layer],self.W_norm_ff_b[layer],self.residual_ff[layer])
             att_delta_in= del_ff_r+new_delta  
             att_delta_X_k,att_delta_X_v,att_delta_X_q,att_delta_W_q,att_delta_W_k,att_delta_W_v,att_delta_W_o,\
             att_delta_W_q_b,att_delta_W_k_b,att_delta_W_v_b,att_delta_W_o_b=self.grad_att(self.dropout['self_att_out'].backward(att_delta_in),self.W_o[layer].weight,self.att_o[layer],self.att_a[layer],self.att_raw_a[layer],self.att_v[layer],self.att_q[layer],self.att_k[layer],self.att_inputs[layer],self.W_q[layer].weight,self.W_k[layer].weight,self.W_v[layer].weight)
             att_delta_out= (att_delta_X_v+att_delta_X_k+att_delta_X_q)
             del_att_r,del_att_alpha,del_att_beta= self.grad_layer_norm_pre(att_delta_out,self.W_norm_att_a[layer],self.W_norm_att_b[layer],self.residual_att[layer])
             new_delta =del_att_r+att_delta_in
             

             layer_grad_store.append({
            'layer': layer,
            # FF
            'ff_w1': ff_w1_grad,   'ff_b1': ff_b1_grad,
            'ff_w2': ff_w2_grad,   'ff_b2': ff_b2_grad,
            # FF norm
            'ff_alpha': del_ff_alpha, 'ff_beta': del_ff_beta,
            # Self attention weights
            'sq': att_delta_W_q,  'sk': att_delta_W_k,
            'sv': att_delta_W_v,  'so': att_delta_W_o,
            # Self attention biases
            'sq_b': att_delta_W_q_b, 'sk_b': att_delta_W_k_b,
            'sv_b': att_delta_W_v_b, 'so_b': att_delta_W_o_b,
            # Attention norm
            'att_alpha': del_att_alpha, 'att_beta': del_att_beta,
        })
            #  self.W_ff1[layer].weight -= self.W_ff1_ada[layer].grad(ff_w1_grad,self.W_ff1[layer].weight)
            #  self.W_ff2[layer].weight -= self.W_ff2_ada[layer].grad(ff_w2_grad,self.W_ff2[layer].weight)
            #             #  self.W_ff1[layer].weight -= 0.001*ff_w1_grad
            #             #  self.W_ff2[layer].weight -= 0.001*ff_w2_grad
            # #  print(abs_mean(self.W_ff2[layer].weight),self.W_ff2_ada[layer].grad(ff_w2_grad,self.W_ff2[layer].weight).abs().mean())
            #             #  print(abs_mean(self.W_ff2[layer].weight),self.W_ff2_ada[layer].grad(ff_w2_grad).abs().mean())
            #  self.W_ff1[layer].bias -= self.W_ff1_ada[layer].grad(ff_b1_grad,self.W_ff1[layer].bias)
            #  self.W_ff2[layer].bias -= self.W_ff2_ada[layer].grad(ff_b2_grad,self.W_ff2[layer].bias)
            #  self.W_q[layer].weight -= self.W_q_ada[layer].grad(att_delta_W_q,self.W_q[layer].weight)
            #  self.W_k[layer].weight -= self.W_k_ada[layer].grad(att_delta_W_k,self.W_k[layer].weight)
            #  self.W_v[layer].weight -= self.W_v_ada[layer].grad(att_delta_W_v,self.W_v[layer].weight)
            #  self.W_o[layer].weight -= self.W_o_ada[layer].grad(att_delta_W_o,self.W_o[layer].weight)
            #  self.W_q[layer].bias -= self.W_q_ada[layer].grad(att_delta_W_q_b,self.W_q[layer].bias)
            #  self.W_k[layer].bias -= self.W_k_ada[layer].grad(att_delta_W_k_b,self.W_k[layer].bias)
            #  self.W_v[layer].bias -= self.W_v_ada[layer].grad(att_delta_W_v_b,self.W_v[layer].bias)
            #  self.W_o[layer].bias -= self.W_o_ada[layer].grad(att_delta_W_o_b,self.W_o[layer].bias)
            #             # #  print(self.W_q_ada[layer].grad(att_delta_W_q).abs().mean())
            #  self.W_norm_att_a[layer] -= self.W_norm_self_a_ada[layer].grad(del_att_alpha,self.W_norm_att_a[layer])
            #  self.W_norm_att_b[layer] -= self.W_norm_self_b_ada[layer].grad(del_att_beta,self.W_norm_att_b[layer])
            #  self.W_norm_ff_a[layer] -= self.W_norm_ff_a_ada[layer].grad(del_ff_alpha,self.W_norm_ff_a[layer])
            #  self.W_norm_ff_b[layer] -= self.W_norm_ff_b_ada[layer].grad(del_ff_beta,self.W_norm_ff_b[layer])

             
               # carry forward only what's needed
         new_delta= self.dropout['emb'].backward(new_delta)
         flat_tokens = torch.tensor(tags, device=device).view(-1)          # (B*T,)
         flat_delta  = new_delta.view(-1, self.d_model)                      # (B*T, d_model)
         W_voc_grad  = torch.zeros(self.voc_size, self.d_model, device=device)
         W_voc_grad.index_add_(0, flat_tokens, flat_delta)
         all_grads = [W_voc_grad]
         for store in layer_grad_store:
                all_grads += [
                    store['ff_w1'],  store['ff_b1'],
                    store['ff_w2'],  store['ff_b2'],
                    store['ff_alpha'], store['ff_beta'],
                    store['sq'],  store['sk'],  store['sv'],  store['so'],
                    store['sq_b'], store['sk_b'], store['sv_b'], store['so_b'],
                    store['att_alpha'], store['att_beta'],
                ]

        #  coef = self.clip_grad_norm(all_grads, max_norm)   # ← single global coef for encoder
         self.grads= layer_grad_store
                # ─────────────────────────────────────────
                # PHASE 3 — apply clipped updates
                # ─────────────────────────────────────────
        #  for store in layer_grad_store:
        #             layer = store['layer']
        #             with torch.no_grad():
        #                 # FF weights
        #                 self.W_ff1[layer].weight -= self.W_ff1_ada[layer].grad(store['ff_w1'] * coef, self.W_ff1[layer].weight)
        #                 self.W_ff2[layer].weight -= self.W_ff2_ada[layer].grad(store['ff_w2'] * coef, self.W_ff2[layer].weight)
        #                 self.W_ff1[layer].bias   -= self.W_ff1_ada[layer].grad(store['ff_b1'] * coef, self.W_ff1[layer].bias)
        #                 self.W_ff2[layer].bias   -= self.W_ff2_ada[layer].grad(store['ff_b2'] * coef, self.W_ff2[layer].bias)
        #                 # FF norm
        #                 self.W_norm_ff_a[layer]  -= self.W_norm_ff_a_ada[layer].grad(store['ff_alpha'] * coef, self.W_norm_ff_a[layer])
        #                 self.W_norm_ff_b[layer]  -= self.W_norm_ff_b_ada[layer].grad(store['ff_beta']  * coef, self.W_norm_ff_b[layer])
        #                 # Self attention weights
        #                 self.W_q[layer].weight -= self.W_q_ada[layer].grad(store['sq'] * coef, self.W_q[layer].weight)
        #                 self.W_k[layer].weight -= self.W_k_ada[layer].grad(store['sk'] * coef, self.W_k[layer].weight)
        #                 self.W_v[layer].weight -= self.W_v_ada[layer].grad(store['sv'] * coef, self.W_v[layer].weight)
        #                 self.W_o[layer].weight -= self.W_o_ada[layer].grad(store['so'] * coef, self.W_o[layer].weight)
        #                 # Self attention biases
        #                 self.W_q[layer].bias -= self.W_q_ada[layer].grad(store['sq_b'] * coef, self.W_q[layer].bias)
        #                 self.W_k[layer].bias -= self.W_k_ada[layer].grad(store['sk_b'] * coef, self.W_k[layer].bias)
        #                 self.W_v[layer].bias -= self.W_v_ada[layer].grad(store['sv_b'] * coef, self.W_v[layer].bias)
        #                 self.W_o[layer].bias -= self.W_o_ada[layer].grad(store['so_b'] * coef, self.W_o[layer].bias)
        #                 # Attention norm
        #                 self.W_norm_att_a[layer] -= self.W_norm_self_a_ada[layer].grad(store['att_alpha'] * coef, self.W_norm_att_a[layer])
        #                 self.W_norm_att_b[layer] -= self.W_norm_self_b_ada[layer].grad(store['att_beta']  * coef, self.W_norm_att_b[layer])

        #  counts = torch.zeros(self.voc_size, device=device)
        #  counts.index_add_(0, flat_tokens, torch.ones(flat_tokens.shape[0], device=device))
        #  counts = counts.clamp(min=1)
        #  W_voc_grad = W_voc_grad / counts.unsqueeze(1)
        #  rows_list   = flat_tokens.unique().tolist()
        #  emb_grad=self.emb_ad.grad_emb(W_voc_grad[rows_list] ,rows_list,self.emb.weight[rows_list])
        #  pos_grad= self.pos_ada.grad(torch.sum(new_delta , dim=0),self.pos.weight)
        #  self.emb.weight[rows_list] -= emb_grad
        #  self.pos.weight -= pos_grad
        #  print(self.pos.weight.abs().mean(),pos_grad.abs().mean())
         # FIX 4: always clear cached activations at the end of back()
         #        so they don't accumulate alongside the new ones from the next fit()
        #  self.clear_memory()
         return W_voc_grad,torch.sum(new_delta , dim=0),all_grads
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
     return del_x, del_alpha, del_beta
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
    def grad_ff(self, delta, layer_1_output, ff_w_2, ff_w_1, x):
        # ff_w2 gradient: sum over batch of (layer_1_output[B].T @ delta[B])
        # Old: for B in range(self.batch_size): delta_ff_w2 += layer_1_output[B].T @ delta[B]
        # delta_ff_w2 = torch.einsum('bti,btj->ij', layer_1_output, delta).T
        delta_ff_w2 = (layer_1_output.view(-1, self.d_ff).T @ delta.view(-1, self.d_model)).T
        
        delta_ff_b2 = delta.sum(dim=(0, 1))
        
        # Backprop through W_ff2
        delta_h = delta @ ff_w_2.T
        delta_h = self.dropout['ff_layer_1'].backward(delta_h)
        # ReLU backward
        delta_a = delta_h * (layer_1_output > 0).float()
        
        delta_ff_b1 = delta_a.sum(dim=(0, 1))
        
        # ff_w1 gradient: sum over batch of (x[B].T @ delta_a[B])
        # Old: for B in range(self.batch_size): delta_ff_w1 += x[B].T @ delta_a[B]
        # delta_ff_w1 = torch.einsum('bti,btj->ij', x, delta_a).T
        delta_ff_w1 = (x.view(-1, self.d_model).T @ delta_a.view(-1, self.d_ff)).T
        delta_x = delta_a @ ff_w_1.T
        
        return delta_ff_w1, delta_ff_b1, delta_ff_w2, delta_ff_b2, delta_x
    
    def grad_att(self,delta,W_o,O,A,raw_A,V,Q,K,X,W_q,W_k,W_v):
        B, T, _ = delta.shape
        delta_o = delta @ W_o.T
        # delta_W_o = torch.einsum('bti,btj->ij', O, delta)
        delta_W_o = (O.view(-1, self.d_model).T @ delta.view(-1, self.d_model))
        delta_W_o_b= delta.sum(dim=(0,1))
        heads_delta_o = delta_o.view(B, T, self.h_count, self.d_k).transpose(1, 2)
        heads_delta_A= heads_delta_o@V.transpose(-2,-1)
        heads_delta_A= self.dropout['self_att_a'].backward(heads_delta_A)
        heads_delta_V= A.transpose(-2,-1)@heads_delta_o
        dot_delA_A= (heads_delta_A * raw_A).sum(dim=-1, keepdim=True)
        heads_delta_S = raw_A* (heads_delta_A - dot_delA_A)
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
        delta_W_q= (X.view(-1, self.d_model).T @ heads_delta_Q.view(-1, self.d_model))
        delta_W_k= (X.view(-1, self.d_model).T @ heads_delta_K.view(-1, self.d_model))
        delta_W_v= (X.view(-1, self.d_model).T @ heads_delta_V.view(-1, self.d_model))
        delta_W_q_b=  heads_delta_Q.sum(dim=(0,1))
        delta_W_k_b = heads_delta_K.sum(dim=(0,1))
        delta_W_v_b=  heads_delta_V.sum(dim=(0,1))
        return delta_X_K,delta_X_V,delta_X_Q,delta_W_q,delta_W_k,delta_W_v,delta_W_o,delta_W_q_b,delta_W_k_b,delta_W_v_b,delta_W_o_b

    def self_attention(self,X,layer):
       B, T, _ = X.shape
       Wq= self.W_q[layer].weight
       Wk= self.W_k[layer].weight
       Wv= self.W_v[layer].weight
       Wq_b = self.W_q[layer].bias
       Wk_b = self.W_k[layer].bias
       Wv_b = self.W_v[layer].bias
        # print(Wk.shape)
       Q = (X @ Wq+Wq_b).view(B, T, self.h_count, self.d_k).transpose(1, 2)
       K = (X @ Wk+Wk_b).view(B, T, self.h_count, self.d_k).transpose(1, 2)
       V = (X @ Wv+Wv_b).view(B, T, self.h_count, self.d_k).transpose(1, 2)
       temp=1.5
       
       S = Q @ K.transpose(-2, -1) / (math.sqrt(self.d_k)*temp)        # (B, H, T, T)
    #    S = S - S.max(dim=-1, keepdim=True)[0]
       
       mask = torch.tensor(self.pad_mask_k|self.pad_mask_q, dtype=torch.bool, device=device)
       
       S= S.masked_fill(mask,negative_inf)
       A = F.softmax(S, dim=-1) 
       A = torch.nan_to_num(A, nan=0.0)
       raw_A= A   
       
       A= self.dropout['self_att_a'].train(self.is_training).forward(A)                             # (B, H, T, T)
       O = (A @ V).transpose(1, 2).contiguous().view(B, T, -1)  
    #    print(self.W_o[layer].bias)
       
       Z= O@self.W_o[layer].weight+ self.W_o[layer].bias
       
       return Z, O .clone(), Q .clone(), K .clone(), V .clone(), S, A,raw_A

    
    def layer_norm(self, X, alpha, beta):
        mean = X.mean(dim=-1, keepdim=True)          # ← fix dim
        var  = X.var(dim=-1, keepdim=True, unbiased=False)
        std  = torch.sqrt(var + self.epsilon)
        return (X - mean) / std * alpha + beta
    def update_weights(self,coef):
                 for store in self.grads:
                    layer = store['layer']
                    with torch.no_grad():
                        # FF weights
                        self.W_ff1[layer].weight -= self.W_ff1_ada[layer].grad(store['ff_w1'] * coef, self.W_ff1[layer].weight)
                        self.W_ff2[layer].weight -= self.W_ff2_ada[layer].grad(store['ff_w2'] * coef, self.W_ff2[layer].weight)
                        self.W_ff1[layer].bias   -= self.W_ff1_ada[layer].grad(store['ff_b1'] * coef, self.W_ff1[layer].bias)
                        self.W_ff2[layer].bias   -= self.W_ff2_ada[layer].grad(store['ff_b2'] * coef, self.W_ff2[layer].bias)
                        # FF norm
                        self.W_norm_ff_a[layer]  -= self.W_norm_ff_a_ada[layer].grad(store['ff_alpha'] * coef, self.W_norm_ff_a[layer])
                        self.W_norm_ff_b[layer]  -= self.W_norm_ff_b_ada[layer].grad(store['ff_beta']  * coef, self.W_norm_ff_b[layer])
                        # Self attention weights
                        self.W_q[layer].weight -= self.W_q_ada[layer].grad(store['sq'] * coef, self.W_q[layer].weight)
                        self.W_k[layer].weight -= self.W_k_ada[layer].grad(store['sk'] * coef, self.W_k[layer].weight)
                        self.W_v[layer].weight -= self.W_v_ada[layer].grad(store['sv'] * coef, self.W_v[layer].weight)
                        self.W_o[layer].weight -= self.W_o_ada[layer].grad(store['so'] * coef, self.W_o[layer].weight)
                        # Self attention biases
                        self.W_q[layer].bias -= self.W_q_ada[layer].grad(store['sq_b'] * coef, self.W_q[layer].bias)
                        self.W_k[layer].bias -= self.W_k_ada[layer].grad(store['sk_b'] * coef, self.W_k[layer].bias)
                        self.W_v[layer].bias -= self.W_v_ada[layer].grad(store['sv_b'] * coef, self.W_v[layer].bias)
                        self.W_o[layer].bias -= self.W_o_ada[layer].grad(store['so_b'] * coef, self.W_o[layer].bias)
                        # Attention norm
                        self.W_norm_att_a[layer] -= self.W_norm_self_a_ada[layer].grad(store['att_alpha'] * coef, self.W_norm_att_a[layer])
                        self.W_norm_att_b[layer] -= self.W_norm_self_b_ada[layer].grad(store['att_beta']  * coef, self.W_norm_att_b[layer])
 
    def create_pad_mask_k(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=device)
          # (B, 1, T, 1)
        pad_k = (X == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        return  pad_k # (B, 1, T, T)
    def create_pad_mask_q(self,X):
        if not isinstance(X, torch.Tensor):
          X = torch.tensor(X, device=device)
        pad_q = (X == 0).unsqueeze(1).unsqueeze(3)
        return pad_q
    def get_sinusoidal_positional_encoding(self,seq_len, d_model):
        pos = torch.arange(seq_len).unsqueeze(1)
        i = torch.arange(d_model).unsqueeze(0)

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angles = pos * angle_rates

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])

        return pe.to(device)
    def clear_memory(self):
          self.ff_output.clear()
          self.ff_inputs.clear()
          self.ff_layer_1_output.clear()
          self.att_output.clear()
          self.att_inputs.clear()
          self.att_o.clear()
          self.att_q.clear()
          self.att_k.clear()
          self.att_v.clear()
          self.att_s.clear()
          self.att_a.clear()
          self.att_raw_a.clear()
          self.dropout['emb'].clear()
          self.dropout['self_att_a'].clear()
          
          self.dropout['self_att_out'].clear()
          self.dropout['ff_layer_1'].clear()
          self.dropout['ff_out'].clear()
          self.grads.clear()
    def val(self):
        self.is_training= False
    def train(self):
        self.is_training= True