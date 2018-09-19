import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import config


class RnnEncoder(nn.Module):
    
    def __init__(self, args, padding_idx, src_vocab_size):
        super(RnnEncoder, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(
            src_vocab_size, 
            args.embedding_size, 
            padding_idx = padding_idx
        )
        
        self.rnn = config.RNN_TYPES[args.rnn_type](
            input_size = args.embedding_size, 
            hidden_size = args.hidden_size,
            num_layers = args.num_encoder_layers,
            dropout = args.dropout,
            bidirectional = args.bidirectional
        )
            
    def forward(self, x, hidden):
        pass
    
    def random_init_hidden(self):
        # TODO: use randomly initialized hidden state for beginning of rnn
        # This is only needed for encoder, 
        # since decoder's first hidden state is the output of encoder
        pass
    
    
class RnnDecoder(nn.Module):
    def __init__(self, args, padding_idx, trg_vocab_size):
        super(RnnDecoder, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(
            trg_vocab_size, 
            args.embedding_size, 
            padding_idx = padding_idx
        )
        
        # Use only one layer of RNN in decoder
        self.rnn = config.RNN_TYPES[self.args.rnn_type](
            input_size = args.embedding_size, 
            hidden_size = args.hidden_size,
            dropout = args.dropout
        )
        
        self.linear = nn.Linear(args.hidden_size, trg_vocab_size)

    def forward(self, x, hidden):
        pass
