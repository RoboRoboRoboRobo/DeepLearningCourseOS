from torch import Tensor
from torch import nn
import torch

# TEST UPDATE ON GOOGLE COLAB
BLAHBLAH = 1
class Zaremba(nn.Module):
    def __init__(self, word_vec_size, hidden_size, vocab_size, num_layers, variation='LSTM_no_DO'):
        super(Zaremba, self).__init__()
        ## Optional variations
        self.variation = variation
        self.dropout = 0

        self.LSTM = nn.LSTM(word_vec_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        self.GRU = nn.GRU(word_vec_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        self.FC = nn.Linear(hidden_size, vocab_size)



    def forward(self, x): ## x is an index of a word in a sorted vocab
        if 'DO' in self.variation:
            self.dropout = 0.2 # Paper
        if 'LSTM' in self.variation:
            x, _ = self.LSTM(x)
        elif 'GRU' in self.variation:
            x, _ = self.GRU(x)
        x = self.FC(x)
        return x

class Embedding(nn.Module): ## TODO
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab = nn.Parameter(Tensor(vocab_size, hidden_size, type = torch.float64))

    def forward(self, x):
        return self.vocab[x]
