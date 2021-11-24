from torch import nn

class Zaremba(nn.Module):
    def __init__(self, word_vec_size, hidden_size, num_layers, variation='LSTM_no_DO'):
        super(Zaremba, self).__init__()
        self.variation = variation
        self.dropout = 0
        self.LSTM = nn.LSTM(word_vec_size, hidden_size, num_layers, dropout=self.dropout)
        self.GRU = nn.GRU(word_vec_size, hidden_size, num_layers, dropout=self.dropout)

        ## Optional variations
        self.variation = variation

    def forward(self, x):
        if 'DO' in self.variation:
            self.dropout = 0.2 # Paper
        if 'LSTM' in self.variation:
            x = self.LSTM(x)
        elif 'GRU' in self.variation:
            x = self.GRU(x)
        return x
