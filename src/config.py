import torch.nn as nn

SOS_TOKEN = 1
EOS_TOKEN = 2

RNN_TYPES = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}