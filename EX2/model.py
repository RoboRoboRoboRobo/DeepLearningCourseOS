from torch import Tensor
from torch import nn

class Zaremba(nn.Module):
    def __init__(self, word_vec_size, hidden_size, vocab_size, num_layers, variation='LSTM_no_DO'):
        super(Zaremba, self).__init__()
        self.LSTM = nn.LSTM(word_vec_size, hidden_size, num_layers, dropout=self.dropout)
        self.GRU = nn.GRU(word_vec_size, hidden_size, num_layers, dropout=self.dropout)
        self.FC = nn.Linear(hidden_size, vocab_size)

        ## Optional variations
        self.variation = variation
        self.dropout = 0

    def forward(self, x): ## x is an index of a word in a sorted vocab
        if 'DO' in self.variation:
            self.dropout = 0.2 # Paper
        if 'LSTM' in self.variation:
            x = self.LSTM(x)
        elif 'GRU' in self.variation:
            x = self.GRU(x)
        x = self.FC(x)
        return x

class Embedding(nn.Module): ## TODO
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab = nn.Parameter(Tensor(vocab_size, hidden_size))

    def forward(self, x):
        return self.vocab[x]
