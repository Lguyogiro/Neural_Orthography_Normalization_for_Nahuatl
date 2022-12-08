from models import *
from torch import nn
import torch
from torchtext.data import Field, TabularDataset, BucketIterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

char_tokenize = lambda s: s.split()
SRC = Field(tokenize=char_tokenize, init_token='<sow>', eos_token='<eow>', lower=True)
TGT = Field(tokenize=char_tokenize, init_token='<sow>', eos_token='<eow>', lower=True)
LANG = Field(sequential=False)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TGT.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)


dec = AttentionDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)



model = Seq2SeqMultiTask(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    print("Starting training...")
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.tgt
        optimizer.zero_grad()
        output = model(src, trg, 0.5)  # use teacher forcing during training only.
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)        
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)