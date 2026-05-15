import numpy as np
import json
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

import os

DATASET_NAME   = "opus100"
SRC_LANG       = "en"
TGT_LANG       = "fr"
DATASET_CONFIG = f"{SRC_LANG}-{TGT_LANG}"
VOCAB_SIZE=16000
BOS_ID=2
EOS_ID=3
def build_tokenizer(dataset, save_path: str, vocab_size: int = VOCAB_SIZE):
    """Train a shared BPE tokenizer on source + target sentences."""
    if os.path.exists(save_path):
        print(f"Loading tokenizer from {save_path}")
        return Tokenizer.from_file(save_path)

    print("Training BPE tokenizer …")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=2,
    )

    def sentence_iter():
        for item in dataset["train"]:
            t = item["translation"]
            yield t[SRC_LANG]
            yield t[TGT_LANG]
    
    tokenizer.train_from_iterator(sentence_iter(), trainer=trainer)

    # Post-processor: wrap with BOS / EOS automatically
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", BOS_ID), ("[EOS]", EOS_ID)],
    )

    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer

def sentence_iter(dataset):
        for item in dataset["train"]:
            t = item["translation"]
            yield t[SRC_LANG]
            yield t[TGT_LANG]
def create_train_samples(ds):
     train_encoder_input= []
     train_decoder_input= []
     tokenizer=Tokenizer.from_file("bpe_translation.json")
     for item in ds['train']:
          t = item['translation']
          enc_inp= tokenizer.encode(t[SRC_LANG]).ids
          dec_inp= tokenizer.encode(t[TGT_LANG]).ids
          train_encoder_input.append(enc_inp)
          train_decoder_input.append(dec_inp)
     return train_encoder_input,train_decoder_input
def create_validation_samples(ds):
     train_encoder_input= []
     train_decoder_input= []
     tokenizer=Tokenizer.from_file("bpe_translation.json")
     for item in ds['validation'].select(range(2000)):
          t = item['translation']
          enc_inp= tokenizer.encode(t[SRC_LANG]).ids
          dec_inp= tokenizer.encode(t[TGT_LANG]).ids
          train_encoder_input.append(enc_inp)
          train_decoder_input.append(dec_inp)
     return train_encoder_input,train_decoder_input
def manual_pad_and_truncate(inputs, max_len, eos_id, pad_id=0):
    # 1. TRUNCATION LOGIC
    # If sequence is too long, cut it to (max_len - 1) to leave space for EOS
   outputs=[]
   for seq in inputs:
    if len(seq) >= max_len:
        new_seq = seq[:max_len-1] + [eos_id]
    
    # 2. PADDING LOGIC
    # If sequence is short, add EOS then fill remaining space with pad_id
    else:
        new_seq = seq 
        padding_needed = max_len - len(new_seq)
        
        new_seq += [pad_id] * padding_needed

    outputs.append( new_seq)
   return outputs
def create_decoder_inputs_targets(sequences, max_len, pad_id=0, bos_id=2, eos_id=3):
    decoder_inputs = []
    decoder_targets = []

    for seq in sequences:
        # Ensure BOS & EOS
        if seq[0] != bos_id:
            seq = [bos_id] + seq
        if seq[-1] != eos_id:
            seq = seq + [eos_id]

        # Shift
        dec_inp = seq[:-1]
        dec_tgt = seq[1:]

        # Truncate
        dec_inp = dec_inp[:max_len]
        dec_tgt = dec_tgt[:max_len]

        # Pad
        dec_inp += [pad_id] * (max_len - len(dec_inp))
        dec_tgt += [pad_id] * (max_len - len(dec_tgt))

        decoder_inputs.append(dec_inp)
        decoder_targets.append(dec_tgt)

    return decoder_inputs, decoder_targets

if __name__ == "__main__":
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    print(ds.keys())
    train_encoder_input_raw,train_decoder_input_raw=create_train_samples(ds)
    with open("./translation_data/raw_data/train_encoder_input.json","w") as file:
      json.dump(train_encoder_input_raw,file)
    with open("./translation_data/raw_data/train_decoder_input.json","w") as file:
      json.dump(train_decoder_input_raw,file)
    
    # build_tokenizer(ds,'./bpe_translation.json',VOCAB_SIZE)
    i=0
    print(train_decoder_input_raw[0])
    # train_encoder_input=manual_pad_and_truncate(train_encoder_input_raw,512,3)
    train_encoder_input = manual_pad_and_truncate(
    train_encoder_input_raw, 128,3
    )

    train_decoder_input, train_decoder_target = create_decoder_inputs_targets(
        train_decoder_input_raw, 128
    )
    print(train_decoder_input_raw[0])
    with open("./translation_data/train/train_encoder_input.json","w") as file:
      json.dump(train_encoder_input,file)
    with open("./translation_data/train/train_decoder_input.json","w") as file:
      json.dump(train_decoder_input,file)
    with open("./translation_data/train/train_decoder_target.json","w") as file:
      json.dump(train_decoder_target,file)
    validation_encoder_input_raw,validation_decoder_input_raw=create_validation_samples(ds)
    with open("./translation_data/raw_data/validation_encoder_input.json","w") as file:
      json.dump(validation_encoder_input_raw,file)
    with open("./translation_data/raw_data/validation_decoder_input.json","w") as file:
      json.dump(validation_decoder_input_raw,file)
    
    # build_tokenizer(ds,'./bpe_translation.json',VOCAB_SIZE)
    i=0
    print(validation_decoder_input_raw[0])
    # validation_encoder_input=manual_pad_and_truncate(validation_encoder_input_raw,512,3)
    validation_encoder_input = manual_pad_and_truncate(
    validation_encoder_input_raw, 128,3
    )

    validation_decoder_input, validation_decoder_target = create_decoder_inputs_targets(
        validation_decoder_input_raw, 128
    )
    print(validation_decoder_input_raw[0])
    with open("./translation_data/validation/validation_encoder_input.json","w") as file:
      json.dump(validation_encoder_input,file)
    with open("./translation_data/validation/validation_decoder_input.json","w") as file:
      json.dump(validation_decoder_input,file)
    with open("./translation_data/validation/validation_decoder_target.json","w") as file:
      json.dump(validation_decoder_target,file)
    
    # for sent in validation_encoder_input:
    #     print(len(sent))

    # for sent in sentence_iter(ds):

    #     print(sent)
    #     if( i==100):
    #          break
    #     i+=1
    # Basic stats
    # print("Total samples:", len(lengths))
    # print("Max length:", lengths.max())
    # print("Min length:", lengths.min())
    # print("Mean length:", lengths.mean())
    # print("Median length:", np.median(lengths))
    # print("Std deviation:", lengths.std())

    # # Percentiles (important for choosing block size)
    # print("90th percentile:", np.percentile(lengths, 90))
    # print("95th percentile:", np.percentile(lengths, 95))
    # print("99th percentile:", np.percentile(lengths, 99))

    # # Longest & shortest sentences
    # max_idx = np.argmax(lengths)
    # min_idx = np.argmin(lengths)

    # print("\nLongest sentence:")
    # print(ds['train'][max_idx]['translation'][SRC_LANG])

    # print("\nShortest sentence:")
    # print(ds['train'][min_idx]['translation'][SRC_LANG])

    # # Histogram (coarse distribution)
    # hist, bins = np.histogram(lengths, bins=80)
    # print("\nHistogram (counts per bin):")
    # for i in range(len(hist)):
    #     print(f"{int(bins[i])}-{int(bins[i+1])}: {hist[i]}")
    
     # with open("validate_tokens.json",'rb') as file:
     #      validate_tokens= json.load(file)
     
     # encoder_input, decoder_input,decoder_target= create_blocks(validate_tokens)
     # np.save('validate_encoder',encoder_input)
     # np.save('validate_decoder_input',decoder_input)
     # np.save('validate_decoder_target',decoder_target)  
     # with open("train_tokens.json",'rb') as file:
     #      validate_tokens= json.load(file)
     
     # encoder_input, decoder_input,decoder_target= create_blocks(validate_tokens)
     # np.save('train_encoder',encoder_input)
     # np.save('train_decoder_input',decoder_input)
     # np.save('train_decoder_target',decoder_target) 
    samples = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I would like to order a coffee.",
        "The quick brown fox jumps over the lazy dog.",
         ]
    tokenizer=Tokenizer.from_file("bpe_translation.json")
    print(tokenizer.encode(samples[3]).ids+[0]*(128-len(tokenizer.encode(samples[3]))))
    print(len([2, 421, 6505, 2026, 1300, 230, 2100, 329, 410, 1603, 1073, 226, 250, 5655, 8265, 17, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
     

