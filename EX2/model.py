from torch import Tensor
from torch import nn
import torch

class Embedding(nn.Module):
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
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = Embedding(vocab_size, hidden_size)
        self.lstms = [nn.LSTM(hidden_size, hidden_size) for i in range(num_layers)]
        self.grus = [nn.GRU(hidden_size, hidden_size) for i in range(num_layers)]
        self.FC = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.4)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.055, 0.055)  ## According to paper increased as hidden size decrease to 200

    def state_init(self, batch_size, device):
        dev = next(self.parameters()).device
        if 'LSTM' in self.variation:
            states = [(torch.zeros(1, batch_size, layer.hidden_size, device = device),
                       torch.zeros(1, batch_size, layer.hidden_size, device = device))
                       for layer in self.lstms]
        if 'GRU' in self.variation:
            states = [(torch.zeros(1, batch_size, layer.hidden_size, device = device),
                       torch.zeros(1, batch_size, layer.hidden_size, device = device))
                       for layer in self.grus]
        return states

    def detach(self, states):
        return [(h.detach(), c.detach()) for (h,c) in states]

    def forward(self, x, states):  ## x is an index of a word in a sorted vocab
        x = self.embed(x)
        x = self.dropout(x)
        if 'LSTM' in self.variation:
            for i, rnn in enumerate(self.lstms):
                x, states[i] = rnn(x, states[i])
                if 'DO' in self.variation:
                    x = self.dropout(x)
        if 'GRU' in self.variation:
            for i, rnn in enumerate(self.grus):
                x, states[i] = rnn(x, states[i])
                if 'DO' in self.variation:
                    x = self.dropout(x)
        scores = self.FC(x)
        return scores, states


