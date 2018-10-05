import torch.nn as nn

SOS_TOKEN = 2
EOS_TOKEN = 3

RNN_TYPES = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}