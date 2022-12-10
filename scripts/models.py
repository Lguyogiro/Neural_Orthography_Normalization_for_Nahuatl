import random
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):        
        embedded = self.dropout(self.embedding(src))     
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #
        # Repeat decoder hidden state src_len times in order to concatenate it 
        # with the encoder outputs.
        #
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # energy shape: [batch size, src len, dec hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        # attention shape: [batch size, src len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs) 
        a = a.unsqueeze(1)
                
        #
        # Get weighted sum of encoder states (weighted by attention vector)
        #
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
                
        rnn_input = torch.cat((embedded, weighted), dim = 2)            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #
        # Also feed the input embedding and the attended encoder representation 
        # to the fully connected output layer.
        #
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
                
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        
        for t in range(1, trg_len):

            output, hidden, _ = self.decoder(input, hidden, encoder_outputs) 

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            input = trg[t]

            if teacher_forcing_ratio > 0:
                teacher_force = random.random() < teacher_forcing_ratio
                input = trg[t] if teacher_force else top1
            else:
                input = trg[t]

        return outputs, None

class Seq2SeqMultiTask(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.classification_layer1 = nn.Linear(self.encoder.fc.out_features, 256)
        self.classification_layer2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

        self.decoder = decoder
        self.device = device
        
    def call_classifier(self, inp):
        x = self.relu(self.classification_layer1(inp))
        return self.classification_layer2(x)
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.25):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
        lang_id_pred = self.call_classifier(hidden)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        
        for t in range(1, trg_len):

            output, hidden, _ = self.decoder(input, hidden, encoder_outputs) 

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            input = trg[t]

            if teacher_forcing_ratio > 0:
                teacher_force = random.random() < teacher_forcing_ratio
                input = trg[t] if teacher_force else top1
            else:
                input = trg[t]

        return outputs, lang_id_pred
