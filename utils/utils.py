import json
import numpy as np
import os

encoder_len = 64
decoder_len = 64
total_block = encoder_len + decoder_len
stride = 64

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

def create_training_files(filename,dataset):
    print(filename)
    with open(filename,'w',encoding='utf-8') as file:
            for line in dataset:
                  line= line.strip()

                  if(len(line)==0 or line[0]=='='):
                        continue
                  else:
                         file.write(line+'\n')


def create_tokens_from_file(source_file,dest_file,tokenizer):
    tokens=[]
    with open(source_file,'r',encoding='utf-8') as f:
        for line in f:
            encoded=tokenizer.encode(line)
            token_ids=encoded.ids
            tokens.append(token_ids)

    flatten_tokens=[x for xs in tokens for x in xs]
    file_path = dest_file

    with open(file_path, 'w') as file:
        json.dump(flatten_tokens, file, indent=4)
    return flatten_tokens

def get_tokens_from_file(source_file):
      tokens=[]
      with open(source_file,'r',encoding='utf-8') as f:
          tokens=json.load(f)
      return tokens
def make_samples(tokens, seq_len,x_file_path,y_file_path):
    X, Y = [], []
    for i in range(0, len(tokens) - seq_len):
        X.append(tokens[i:i+seq_len])
        Y.append(tokens[i+1:i+seq_len+1])
    with open(x_file_path, 'w') as file:
        json.dump(X, file, indent=4)
    with open(y_file_path, 'w') as file:
        json.dump(Y, file, indent=4)
    return X, Y
def create_blocks(token_ids):
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []

    for i in range(0, len(token_ids) - total_block, stride):

        block = token_ids[i : i + total_block]

        enc = block[:encoder_len]
        dec_target = block[encoder_len:]

        # Append EOS at end of decoder target
        dec_target = dec_target[:-1] + [EOS_ID]

        # Shifted decoder input with BOS
        dec_input = [BOS_ID] + dec_target[:-1]

        encoder_inputs.append(enc)
        decoder_inputs.append(dec_input)
        decoder_targets.append(dec_target)

    return (
        np.array(encoder_inputs),
        np.array(decoder_inputs),
        np.array(decoder_targets)
    )



import gc
import torch

def count_tensors():
    objs = gc.get_objects()
    tensor_count = 0
    total_mem = 0

    for obj in objs:
        try:
            if torch.is_tensor(obj):
                tensor_count += 1
                total_mem += obj.numel() * obj.element_size()
        except:
            pass

    print(f"Tensors: {tensor_count}, Memory: {total_mem / 1024**2:.2f} MB")

import gc
import torch

def debug_top_tensors(n=15):
    tensors = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                size = obj.numel() * obj.element_size()
                tensors.append((size, obj))
        except:
            pass

    tensors.sort(key=lambda x: x[0], reverse=True)

    print("\n===== TOP TENSORS =====")
    print(len(tensors))
    total_leak=0
    for size, t in tensors:
        if(t.requires_grad==True and t.grad_fn):
            total_leak+=size/1024**2
            print(
                f"{size/1024**2:.2f} MB | shape={tuple(t.shape)} | "
                f"req_grad={t.requires_grad} | "
                f"grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}"
            )
    print(total_leak)
def abs_mean(tensor):
    return tensor.abs().mean()