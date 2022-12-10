import time
from models import *
from torch import nn
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
from torch.nn import BCEWithLogitsLoss 
from torch.nn import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH_TO_DATA = "../data/"
USE_SUPPLEMENTAL_SPANISH_DATA = True
BATCH_SIZE = 64 


char_tokenize = lambda s: s.split()
SRC = Field(tokenize=char_tokenize, init_token='<sow>', eos_token='<eow>', lower=True)
TGT = Field(tokenize=char_tokenize, init_token='<sow>', eos_token='<eow>', lower=True)
LANG = Field(sequential=False)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TGT.vocab)
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

def get_data_iterators(use_supplemental_spanish_data=False):
    path_to_data = PATH_TO_DATA
    train_data, dev_data, test_data = TabularDataset.splits(
            path=path_to_data, 
            train=(
                "train_w_xtra_spa_ALL.tsv" if USE_SUPPLEMENTAL_SPANISH_DATA 
                else "train.tsv"
                ), 
            validation="dev.tsv", 
            test="test.tsv",
            format='tsv',
            fields=[('src', SRC), ('tgt', TGT), ('lang', LANG)])

    # If your dataset is huge and contains many unique words, then for the sake of fast execution, you can add this argument: min_freq = <some integer>
    # Only tokens that appear atleast <some integer> times then are considered. Other such words are replaced by < UNK >

    SRC.build_vocab(train_data)
    TGT.build_vocab(train_data)
    LANG.build_vocab(train_data)

    print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target vocabulary: {len(TGT.vocab)}")

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_size=BATCH_SIZE,
        device=device,
        sort_key=lambda x: len(x.src) # batch by length in order to minimize sequence padding
    )
    return train_iterator, valid_iterator, test_iterator, test_data # to get res per sample


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def init_model():
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = AttentionDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2SeqMultiTask(enc, dec, device).to(device)
    model.apply(init_weights)
    return model


def train(model, iterator, optimizer, criterion, clip, 
          classification_criterion=None):
    model.train()
    epoch_loss = 0
    print("Starting training...")
    for batch in iterator:
        
        src = batch.src
        trg = batch.tgt
        lang = batch.lang - 1

        optimizer.zero_grad()
        output, lang_pred = model(src, trg, 0.5)  # use teacher forcing during training only.
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)

        if classification_criterion is not None:
            classification_loss = classification_criterion(
                lang_pred, 
                lang.float().unsqueeze(1)
            )
            total_loss = loss + classification_loss
        else:
            total_loss = loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += total_loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, classification_criterion=None):
    model.eval()
    epoch_loss = 0    
    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.tgt
            lang = batch.lang - 1

            output, lang_pred = model(src, trg, 0) # turn off teacher forcing   
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)

            if classification_criterion is not None:
                classification_loss = classification_criterion(
                    lang_pred, lang.float().unsqueeze(1)
                )
                total_loss = loss + classification_loss
            else:
                total_loss = loss
            epoch_loss += total_loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def translate_sentence(sentence, src_field, trg_field, model, device, 
                       max_len=50):
    model.eval()
    tokens = [token.lower() for token in sentence]
    # Add start of sentence and end of sentence to the tokens
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    # Numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # src_indexes shape = [src_len]1919 batches
    # Convert to tensor and add batch dimension.
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    # Feed source sentence into encoder
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    # create a list to hold the output sentence, initialized with an <sos> token
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    # create a tensor to hold the attention values
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, attention = model.decoder(
                trg_tensor, hidden, encoder_outputs
            )
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

def get_accuracy(data_to_eval):
    total, correct = 0, 0
    preds = []
    inc = []
    for i in range(len(data_to_eval.examples)):
        src = vars(data_to_eval.examples[i])['src']
        trg = vars(data_to_eval.examples[i])['tgt']
        lang = vars(data_to_eval.examples[i])['lang']

        translation, attention = translate_sentence(
            src, SRC, TGT, model, device
        )

        preds.append([src, trg, lang, translation])
        if i % 100 == 0:
            print(i)
        if translation[:-1] == trg:
            correct += 1
        else:
            inc.append((src, trg, translation[:-1]))

        total += 1

    print(f"Acc: {correct / total }")


def main(model, train_iterator, dev_iterator, test_data, multitask=False, 
         n_epochs=100, lr=0.001):
    CLIP = 1
    MULTITASK = False
    optimizer = optim.Adam(model.parameters(), lr=lr)
    TRG_PAD_IDX = TGT.vocab.stoi[TGT.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    if multitask is True:
        classification_criterion = BCEWithLogitsLoss()
    else:
        classification_criterion = None

    train_losses, val_losses = [], []

    best_valid_loss = float('inf')

    print("{} batches".format(len(train_iterator)))
    for epoch in range(n_epochs):
        start_time = time.time()
        
        train_loss = train(
            model, train_iterator, 
            optimizer, criterion, CLIP, 
            classification_criterion=classification_criterion
        )

        valid_loss = evaluate(
            model, dev_iterator, criterion, 
            classification_criterion=classification_criterion
        )

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'MTT={}_xtra_spa_data_model.pt'.format(MULTITASK))
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    print("Acc: {}".format(get_accuracy(test_data)))

    
if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--multitask", "-m", default=False, type=bool)
    args = argparser.parse_args()
    
    train_iter, dev_iter, test_iter, test_data = get_data_iterators(args.multitask)
    model = init_model()
    main(model, train_iter, dev_iter, test_data, multitask=False)