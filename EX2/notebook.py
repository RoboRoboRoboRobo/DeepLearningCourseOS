## in notebook
from model import Zaremba, Embedding ## move to notebook
from preprocess import preprocess_ptb_files, create_dataset
import torch
word_vec_size = 200
vocab_size = 10000 ## need to be calculated
num_layers = 2
batch_size = 20
sequence_length = 35
lr = 0.001
epoch_num = 100
variation = 'LSTM_no_DO'
checkpoints_dir_path = '/Users/shir.barzel/DeepLearningCourseOS/EX2/results'
trn_data, val_data, tst_data = preprocess_ptb_files('/Users/shir.barzel/DeepLearningCourseOS/EX2/PTB/ptb.char.train.txt',
                     '/Users/shir.barzel/DeepLearningCourseOS/EX2/PTB/ptb.char.valid.txt',
                     '/Users/shir.barzel/DeepLearningCourseOS/EX2/PTB/ptb.char.test.txt') ## TODO

trn_dataset = create_dataset(trn_data, batch_size, sequence_length)
val_dataset = create_dataset(val_data, batch_size, sequence_length)
tst_dataset = create_dataset(tst_data, batch_size, sequence_length)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Zaremba(word_vec_size, word_vec_size, vocab_size, num_layers, variation)
embed = Embedding(vocab_size, word_vec_size)
optimizer = optim.Adam(params = model.parameters(), lr=lr, weight_decay=0)

events_dir = assignment_path + '/events/'
writer = SummaryWriter(events_dir)

model.to(device)
train(model, trn_dataset[:5], val_dataset[:5], tst_dataset[:5],
      embed, device, variation, optimizer, epoch_num, checkpoints_dir_path, writer) # TODO remove 5
## in notebook

