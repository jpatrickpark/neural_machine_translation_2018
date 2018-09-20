import argparse
import loader
from torchtext import data
import models
from torch.nn import init
import config
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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
        for _, param in net.named_parameters(): 
            init.normal_(param, std=0.01)

    # TODO: other optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.l2_penalty)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.l2_penalty)
    criterion = nn.NLLLoss()
    
    # TODO: save/load weights
    # TODO: early stopping
    for i in range(args.epoch):
        if args.test:
            test(args, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, i, test_data)
        else:
            train_and_val(args, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, i, train_data, val_data)
        
def train_model(args, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch, device):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    loss = 0
    
    # start decoder input by config.SOS_TOKEN
    # stop when config.EOS_TOKEN is detected
    # TODO: implement this part using F.softmax
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    # return loss
    return None
    
    
def train_and_val(args, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, epoch_idx, train_data, val_data):
    train_iter, val_iter = data.BucketIterator.splits(
        (train_data, val_data), batch_size=args.batch_size
    )

    for i, train_batch in enumerate(iter(train_iter)):
        loss = train_model(
            args,
            encoder,
            decoder, 
            encoder_optimizer, 
            decoder_optimizer, 
            criterion,
            train_batch,
            device
        )
        if i % args.print_every == 0:
            print("epoch: {}, step: {}, loss: {}".format(epoch_idx, i, loss))
        
    for val_batch in iter(val_iter):
        pass
        
def test():
    pass

def rnn_encoder_decoder_argparser():
    # TODO: set min max size for vocab
    # TODO: specify optimizer
    # TODO: set max length of sentences
    parser = argparse.ArgumentParser(description='Run Tests')
    parser.add_argument('--test', help="Run test instead of training/validation", action="store_true")
    parser.add_argument("--njobs", help="Number of jobs to use when loading data", type=int, default=1)
    parser.add_argument("--epoch", help="Number of epoch to run", type=int, default=1)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=16)
    parser.add_argument("--hidden_size", help="Hidden size", type=int, default=128)
    parser.add_argument("--embedding_size", help="Embedding size", type=int, default=128)
    parser.add_argument("--print_every", help="How frequently print result", type=int, default=1000)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=0.001)
    parser.add_argument("--dropout", help="Dropout Rate", type=float, default=0.2)
    parser.add_argument('--bidirectional', help="Use bidirectional RNN in encoder", action="store_true") # don't set this true in this model
    parser.add_argument('--cpu', help="Use cpu instead of gpu", action="store_true")
    parser.add_argument("--data", help="Directory where data is stored", default='../data/neu2017/')
    parser.add_argument("--rnn_type", help="Which rnn to use (rnn, lstm, gru)", default='gru')
    parser.add_argument("--num_encoder_layers", help="Number of rnn layers in encoder", type=int, default=1)
    parser.add_argument("--early_stopping", help="Stop if validation does not improve", type=int, default=5)
    parser.add_argument('--l2_penalty', help="L2 pelnalty coefficient in optimizer", type=float,  default=0.001)
    return parser

if __name__ == '__main__':
    parser = rnn_encoder_decoder_argparser()
    args = parser.parse_args()
    
    run(args)
    

