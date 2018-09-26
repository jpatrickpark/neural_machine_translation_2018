import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import config


class RnnEncoder(nn.Module):
    
    def __init__(self, args, padding_idx, src_vocab_size):
        super(RnnEncoder, self).__init__()
        self.args = args
        self.num_directions = 2 if args.bidirectional else 1

        self.embedding = nn.Embedding(
            src_vocab_size, 
            args.embedding_size, 
            padding_idx = padding_idx
        )
        
        self.rnn_type = args.rnn_type
        
        self.rnn = config.RNN_TYPES[args.rnn_type](
            input_size = args.embedding_size, 
            hidden_size = args.hidden_size,
            num_layers = args.num_encoder_layers,
            dropout = args.dropout,
            bidirectional = args.bidirectional
        )
            
    def forward(self, x):
        #print("src shape", x.shape) # torch.Size([32, 16])
        # dimenstion of x: (seq_len, batch, input_size)
        x = self.embedding(x)
        #print("embedded shape", x.shape) # torch.Size([32, 16, 256])
        # dimension of x after embedding: (seq_len, batch, embedding_size)
        if self.rnn_type == 'lstm':
            x, (self.hidden, self.cell_state) = self.rnn(x, (self.hidden, self.cell_state))
        else:
            x, self.hidden = self.rnn(x, self.hidden)
        #print("after encoder shape", x.shape) # torch.Size([32, 16, 128])
        # dimension of x after encoder: (seq_len, batch, hidden_size)
        #print("encoder hidden shape", self.hidden.shape) # torch.Size([1, 16, 128])
        return x
    
    def random_init_hidden(self, device, current_batch_size):
        # This is only needed for encoder, 
        # since decoder's first hidden state is the output of encoder
        # LSTM: output, (h_n, c_n)
        # rnn, gru: output, h_n
        
        # we declare initial hidden tensor every time because the last batch size
        # might be different from the rest of the batch size,
        # but feel free to modify this if you have a better idea.
        self.hidden = torch.zeros(
            self.args.num_encoder_layers * self.num_directions, 
            current_batch_size, 
            self.args.hidden_size, 
            device=device
        )
        
        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        nn.init.xavier_normal_(self.hidden)
        
        if self.rnn_type == 'lstm':
            self.cell_state = torch.zeros(
                self.args.num_encoder_layers * self.num_directions, 
                current_batch_size, 
                self.args.hidden_size, 
                device=device
            )
            nn.init.xavier_normal_(self.cell_state)
    
    
class RnnDecoder(nn.Module):
    def __init__(self, args, padding_idx, trg_vocab_size):
        super(RnnDecoder, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(
            trg_vocab_size, 
            args.embedding_size, 
            padding_idx = padding_idx
        )
        
        # Use only one layer of RNN in decoder for now
        self.rnn = config.RNN_TYPES[self.args.rnn_type](
            input_size = args.embedding_size, 
            hidden_size = args.hidden_size,
            dropout = args.dropout
        )
        
        self.linear = nn.Linear(args.hidden_size, trg_vocab_size)

    def forward(self, x):
        #print("trg shape", x.shape) torch.Size([40, 16])
        x = self.embedding(x)
        #print("embedded shape", x.shape) # torch.Size([40, 16, 256])
        # If we pass SOS token here, and run it iterative fashion, then it's translation
        # thus we need to set maximum sequence length
        # if we pass the entire training data here, then it's using teacher forcing.
        # TODO: Do we use relu here?
        x, self.hidden = self.rnn(x, self.hidden) # this is actually using teacher forcing
        #print("after decoder shape", x.shape) # torch.Size([40, 16, 128])
        #print("decoder hidden shape", self.hidden.shape) # torch.Size([1, 16, 128])
        x = self.linear(x)
        #print("after linear shape", x.shape) # torch.Size([40, 16, 5679])
        return x

    
class RnnEncoderDecoder(nn.Module):
    # I thought this will be nicer to use a big model that has both encoder and decoder
    # But I am not sure if this is actually feasible 
    # since we have to reuse output of the decoder as input to decoder,
    # it might be better if decoder is a separate unit.
    def __init__(self, args, padding_idx, src_vocab_size, trg_vocab_size):
        super(RnnEncoderDecoder, self).__init__()
        self.args = args

        self.encoder = RnnEncoder(args, padding_idx, src_vocab_size)
        self.decoder = RnnDecoder(args, padding_idx, trg_vocab_size)
        
        #self.encoder.random_init_hidden()
        
    def forward(self, src, trg):
        print(src.shape)
        print(trg.shape)
        self.encoder(src) # we do not use output of encoder
        output = self.decoder(trg)
        return output