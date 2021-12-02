from torch import Tensor
from torch import nn
import torch

class Embedding(nn.Module): ## TODO
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab = nn.Parameter(Tensor(vocab_size, hidden_size))

    def forward(self, x):
        return self.vocab[x]

class Zaremba(nn.Module):
    def __init__(self, word_vec_size, hidden_size, vocab_size, num_layers, variation='LSTM_no_DO'):
        super(Zaremba, self).__init__()
        ## Optional variations
        self.variation = variation
        self.dropout = 0
        self.embed = Embedding(vocab_size, hidden_size)
        self.LSTM = nn.LSTM(word_vec_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        self.GRU = nn.GRU(word_vec_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        self.FC = nn.Linear(hidden_size, vocab_size)
        self.init_parameters()

    def init_parameters(self): ## TODO only for vocab
        for param in self.parameters():
            nn.init.uniform_(param, -0.05, 0.05) ## TODO winit

    def forward(self, x): ## x is an index of a word in a sorted vocab
        x = self.embed(x)
        if 'DO' in self.variation:
            self.dropout = 0.2 # Paper
        if 'LSTM' in self.variation:
            x, _ = self.LSTM(x)
        elif 'GRU' in self.variation:
            x, _ = self.GRU(x)
        x = self.FC(x)
        return x


