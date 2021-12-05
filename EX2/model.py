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
        self.dropout = nn.Dropout(p=0.5)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = Embedding(vocab_size, hidden_size)
        self.LSTM = nn.LSTM(word_vec_size, hidden_size)
        self.GRU = nn.GRU(word_vec_size, hidden_size)
        self.FC = nn.Linear(hidden_size, vocab_size)
        self.init_parameters()

    def init_parameters(self):  ## TODO only for vocab
        for param in self.parameters():
            nn.init.uniform_(param, -0.055, 0.055)  ## According to paper increased as hidden size decrease to 200

    def state_init(self, batch_size, device):
        dev = next(self.parameters()).device
        states = (torch.zeros(1, batch_size, self.hidden_size, device=device), torch.zeros(1, batch_size, self.hidden_size, device=device))
        return states

    def detach(self, states):
        return (states[0].detach(), states[1].detach())

    def forward(self, x, states):  ## x is an index of a word in a sorted vocab
        x = self.embed(x)
        if 'LSTM' in self.variation:
            for i in range(self.num_layers):
                x, states = self.LSTM(x)
                if 'DO' in self.variation:
                    x = self.dropout(x)
        if 'GRU' in self.variation:
            for i in range(self.num_layers):
                x, states = self.GRU(x, states)
                if 'DO' in self.variation:
                    x = self.dropout(x)
        scores = self.FC(x)
        return scores, states


