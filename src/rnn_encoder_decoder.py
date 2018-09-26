import argparse
import loader
from torchtext import data
import models
#from torch.nn import init
import config
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

def run(args):
    device = torch.device("cuda" if (not args.cpu) and torch.cuda.is_available() else "cpu")
    
    train_data, val_data, test_data, src, trg = loader.load_chinese_english_data(args.data, args.njobs)
    
    src_padding_idx = src.vocab.stoi['<pad>']
    trg_padding_idx = trg.vocab.stoi['<pad>']
    
    #src_unk_idx = src.vocab.stoi['<unk>']
    #trg_unk_idx = trg.vocab.stoi['<unk>']
    
    src_vocab_size = len(src.vocab)
    trg_vocab_size = len(trg.vocab)
    
    encoder = models.RnnEncoder(args, src_padding_idx, src_vocab_size).to(device)
    decoder = models.RnnDecoder(args, trg_padding_idx, trg_vocab_size).to(device)
    
    # initialize weights using gaussian with 0 mean and 0.01 std, just like the paper said
    # TODO: Better initialization. Xavier?
    for net in [encoder, decoder]:
        for name, param in net.named_parameters(): 
            #print(name, type(param), param)
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
            #nn.init.normal_(param, std=0.01)

    # TODO: other optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.l2_penalty)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.l2_penalty)
    
    # TODO: use different loss?
    loss_function = nn.NLLLoss()
    
    # TODO: save/load weights
    # TODO: early stopping
    loss_history = defaultdict(list)
    for i in range(args.epoch):
        if args.test:
            test(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, i, test_data)
        else:
            train_loss, val_loss = train_and_val(args, encoder, decoder, encoder_optimizer, 
                                                 decoder_optimizer, loss_function, device, i, train_data, val_data)
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            if early_stop(loss_history["val"], args.early_stopping):
                print("Early stopped.")
                break
                
def early_stop(loss_history, early_stop_k):
    if len(loss_history) < early_stop_k:
        return False
    idx_min = loss_history.index(min(loss_history))
    return len(loss_history) - 1 - idx_min >= early_stop_k
        
def run_batch(phase, args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, batch, device):

    assert phase in ("train", "val", "test"), "invalid phase"
    if phase == "train":
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
    loss = 0
    
    # start decoder input by config.SOS_TOKEN
    # stop when config.EOS_TOKEN is detected
    # TODO: it seems that currently batch size is always the same. Make sure to use the last batch
    target_sequence_length, batch_size = batch.trg.shape
    #print("This should be batch size:", batch_size) #
    
    encoder.random_init_hidden(device, batch_size)
    
    encoder(batch.src)
    
    # this is needed when we are not using teacher forcing
    decoder_input = torch.tensor([[config.SOS_TOKEN]*args.batch_size], device=device, requires_grad=False)
    
    # This step is necessary to get the hidden state from encoder
    decoder.hidden = encoder.hidden
    
    # TODO: to feed decoder's output word, the vocabulary with the highest probability,
    # into itself when predicting next token in sequence and only use teacher forcing
    # with probability of args.teacher_forcing
    #if (phase == "train") and (np.random.rand() < args.teacher_forcing):
    #    pass
    #else:
    #    pass
    
    # TEACHER FORCING
    # Feed all target sentences at once instead of reusing output as input
    # nice to look, and should be fast
    logits = decoder(batch.trg)
    # get prediction
    # log_softmax with NLLLoss == CrossEntropyLoss with logits
    output = F.log_softmax(logits, dim=1)
    #loss = torch.nn.CrossEntropyLoss()(logits, dim=1)
    # Now, loop through output and calculate loss
    # be careful not to compare SOS_token with first non_sos token
    # make sure to stop when encountered EOS_token
    
    # first output is what the decoder produced after seeing SOS token
    # therefore, compare it with second target token
    # TODO: we do not need to feed the last EOS token to the decoder. fix it
    # TODO: we do not need to feed any padding token either. 
    #       do not calculate loss for these tokens for some shorter batches
    for i in range(target_sequence_length-1):
        loss += loss_function(output[i,:,:], batch.trg[i+1,:])
        
    
    # To feed output of the decoder into itself, has to use for loop, will be slower (is there faster alternative?)

    if phase == "train":
        loss.backward() # this should calculate gradient for both encoder and decoder

        nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
        nn.utils.clip_grad_norm(decoder.parameters(), args.clip)

        decoder_optimizer.step() # does it really matter which one takes step first?
        encoder_optimizer.step()

    # return loss / batch_size to compare loss between batches of different sizes
    return loss.item() / batch_size
    
    
def train_and_val(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, epoch_idx, train_data, val_data):
    train_iter, val_iter = data.BucketIterator.splits(
        (train_data, val_data), batch_size=args.batch_size
    )
    
    train_iter = data.BucketIterator(
        dataset=train_data, 
        batch_size=args.batch_size,
        repeat=False
    )
    
    val_iter = data.BucketIterator(
        dataset=val_data, 
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        #A key to use for sorting examples in order to batch together 
        # examples with similar lengths and minimize padding.
        sort=True,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)),
        repeat=False
    )

    # turn on dropout
    encoder.train()
    decoder.train()
    
    train_loss_list = []
    for i, train_batch in enumerate(iter(train_iter)):
        loss = run_batch(
            "train",
            args,
            encoder,
            decoder, 
            encoder_optimizer, 
            decoder_optimizer, 
            loss_function,
            train_batch,
            device
        )
        train_loss_list.append(loss)
        # TODO: use tensorboard to see train / val plot while training
        if i % args.print_every == 0:
            print("train, epoch: {}, step: {}, average loss for current epoch: {}, batch loss: {}".format(
                epoch_idx, i, np.mean(train_loss_list), loss))
        
    print("train done. epoch: {}, average loss for current epoch: {}, numbatch: {}, size of last batch: {}".format(
        epoch_idx, np.mean(train_loss_list), i+1, train_batch.src.shape[1]))
    
    # turn off dropout
    encoder.eval()
    decoder.eval()
    
    val_loss_list = []
    for i, val_batch in enumerate(iter(val_iter)):
        loss = run_batch(
            "val",
            args,
            encoder,
            decoder, 
            encoder_optimizer, 
            decoder_optimizer, 
            loss_function,
            train_batch,
            device
        )
        val_loss_list.append(loss)
        if i % args.print_every == 0:
            print("val, epoch: {}, step: {}, average loss for current epoch: {}, batch loss: {}".format(
                epoch_idx, i, np.mean(val_loss_list), loss))
        
    print("val done. epoch: {}, average loss for current epoch: {}".format(
        epoch_idx, np.mean(val_loss_list)))
    
    return np.mean(train_loss_list), np.mean(val_loss_list)
        
def test():
    pass

def rnn_encoder_decoder_argparser():
    # TODO: set min max size for vocab
    # TODO: specify optimizer
    # TODO: set max length of sentences
    parser = argparse.ArgumentParser(description='Run Tests')
    parser.add_argument('--test', help="Run test instead of training/validation", action="store_true")
    parser.add_argument("--njobs", help="Number of jobs to use when loading data", type=int, default=1)
    parser.add_argument("--epoch", help="Number of epoch to run", type=int, default=10000)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--hidden_size", help="Hidden size", type=int, default=256)
    parser.add_argument("--embedding_size", help="Embedding size", type=int, default=256)
    parser.add_argument("--print_every", help="How frequently print result", type=int, default=100)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-02)
    parser.add_argument("--dropout", help="Dropout Rate", type=float, default=0)
    parser.add_argument('--bidirectional', help="Use bidirectional RNN in encoder", action="store_true") # don't set this true in this model
    parser.add_argument('--cpu', help="Use cpu instead of gpu", action="store_true")
    parser.add_argument("--data", help="Directory where data is stored", default='../data/neu2017/')
    parser.add_argument("--rnn_type", help="Which rnn to use (rnn, lstm, gru)", default='gru')
    parser.add_argument("--num_encoder_layers", help="Number of rnn layers in encoder", type=int, default=1)
    parser.add_argument("--early_stopping", help="Stop if validation does not improve", type=int, default=10)
    parser.add_argument('--l2_penalty', help="L2 pelnalty coefficient in optimizer", type=float,  default=1e-06) #1e-06
    parser.add_argument('--clip', help="clip coefficient in optimizer", type=float,  default=1)
    parser.add_argument('--teacher_forcing', help="probability of performing teacher forcing", type=float,  default=0.5)
    return parser

if __name__ == '__main__':
    parser = rnn_encoder_decoder_argparser()
    args = parser.parse_args()
    
    run(args)
    

