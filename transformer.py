import json
import time

import numpy as np
from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import Decoder, calculate_validation_loss, predict
from adam import AdamCustom,SchedulerState
from encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Transformer :
    def __init__(self,voc_size,d_model,max_len,d_ff,h_count,layers,batch_size,lr=0.0003,epochs=40):
         self.schedular= SchedulerState(lr,4000,62500)
         self.emb=nn.Embedding(voc_size, d_model ,device=device)
         self.emb_ad = AdamCustom(0.99, voc_size, d_model, 0.01,schedular=self.schedular)
         self.pos= nn.Embedding(max_len,d_model,device=device)
         self.pos_ada = AdamCustom(0.99, max_len, d_model, 0.01,scale=True,schedular=self.schedular)

         
         self.decoder= Decoder(d_model,d_ff,h_count,voc_size,max_len,layers,batch_size,schedular=self.schedular)
         self.encoder= Encoder(d_model,h_count,d_ff,voc_size,max_len,layers,batch_size,schedular=self.schedular)
         nn.init.xavier_normal_(self.emb.weight)
         nn.init.xavier_normal_(self.pos.weight)
        #  nn.init.normal_(self.emb.weight, mean=0, std=d_model ** -0.5)
        #  nn.init.normal_(self.pos.weight, mean=0, std=d_model ** -0.5)
    def fit(self,encoder_inputs,decoder_inputs,targets):
         
         self.decoder.emb=self.emb
         self.decoder.pos=self.pos
         self.encoder.emb= self.emb
         self.encoder.pos= self.pos
         self.decoder.train()
         self.encoder.train()
         E,E_pad_mask = self.encoder.fit_pre(encoder_inputs)
         
         prob = self.decoder.fit_pre(decoder_inputs, E,E_pad_mask)
         
         self.schedular.advance()
            
         del_E,loss,dec_emb_grad,dec_pos_grad = self.decoder.back_pre(targets, prob)
    #      print(f"loss:- {loss} | iteration:- {iteration}")
        #  print(del_E)
         enc_emb_grad,enc_pos_grad=self.encoder.back_pre(del_E, E,encoder_inputs)
         self.emb.weight-= self.emb_ad.grad(dec_emb_grad+enc_emb_grad,self.emb)
         self.pos.weight-= self.pos_ada.grad(dec_pos_grad+enc_pos_grad,self.emb)
         
         return loss
if __name__ == "__main__":
    x_train_decoder = []
    x_train_encoder = []
    x_train_target = []

    with open('translation_data/train/train_decoder_input.json', 'rb') as file:
        x_train_decoder = np.array(json.load(file))
    with open('translation_data/train/train_encoder_input.json', 'rb') as file:
        x_train_encoder = np.array(json.load(file))
    with open('translation_data/train/train_decoder_target.json', 'rb') as file:
        x_train_target = np.array(json.load(file))
    with open('translation_data/validation/validation_decoder_input.json', 'rb') as file:
        x_validation_decoder = np.array(json.load(file))
    with open('translation_data/validation/validation_encoder_input.json', 'rb') as file:
        x_validation_encoder = np.array(json.load(file))
    with open('translation_data/validation/validation_decoder_target.json', 'rb') as file:
        x_validation_target = np.array(json.load(file))

    test_batch=x_train_encoder[2:3]
    # iteration=1
    epoch = 11

    model= Transformer(d_model=256, h_count=8, d_ff=512, voc_size=16000, max_len=128, layers=3, batch_size=64)
    model=  torch.load('./models/transformer-20.pth',weights_only=False) 
    # # print("hello")
    iteration=0
    start_time = time.perf_counter()
    total_iteration= len(x_train_encoder)//64
    
    while epoch!=61:
         total_loss=0
         iteration=1
         for i in range(0,len(x_train_encoder)-64,64):
             with torch.no_grad():
                 loss= model.fit(x_train_encoder[i:i+64],x_train_decoder[i:i+64],x_train_target[i:i+64])
                 total_loss+=loss
                #  print(iteration)
                 iteration+=1
         
            
         end_time= time.perf_counter()
         total_time= end_time-start_time
         print(f"total time:- {total_time:.1f}")
         if(epoch%5==0):
            torch.save(model,f'../workspace/models/transformer-{epoch}.pth')

         print(f"loss:- {total_loss/total_iteration} | epoch:- {epoch}")
         epoch+=1
       
         print(calculate_validation_loss(model.encoder,model.decoder,x_validation_encoder,x_validation_decoder,x_validation_target))
    samples = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I would like to order a coffee.",
        "The quick brown fox jumps over the lazy dog.",
         ]
    tokenizer=Tokenizer.from_file("bpe_translation.json")
    sample=np.array([tokenizer.encode(samples[1]).ids+[0]*(128-len(tokenizer.encode(samples[1])))])
    print(predict(sample,x_train_encoder[2:3],model.encoder,model.decoder))
    print(calculate_validation_loss(model.encoder,model.decoder,x_validation_encoder,x_validation_decoder,x_validation_target))
