from model import Zaremba, Embedding
from preprocess import preprocess_ptb_files, create_dataset
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from train import train
from datetime import datetime
import os

word_vec_size = 200
vocab_size = 10000  ## need to be calculated
num_layers = 2
batch_size = 20
sequence_length = 35
lr = 1
lr_factor = 1.3 # decrease as hidden size decrease
lr_change_epoch = 3
epoch_num = 100
max_grad_norm = 3

"""" paths:
OFIR   C:\\Users\ofir-kr\PycharmProjects\DeepLearningCourseOS\EX2\\"
SHIR   /Users/shir.barzel/DeepLearningCourseOS/EX2/
"""
user = "Shir"

if user == "Ofir":
    slash = "\\"
    ptb_dir = "C:\\Users\\ofir-kr\\PycharmProjects\\DeepLearningCourseOS\\EX2\PTB\\"
    checkpoints_dir_path = 'C:\\Users\ofir-kr\\PycharmProjects\\DeepLearningCourseOS\\EX2\\results\\'
elif user == "Shir":
    ptb_dir = "/Users/shir.barzel/DeepLearningCourseOS/EX2/PTB/"
    checkpoints_dir_path = '/Users/shir.barzel/DeepLearningCourseOS/EX2/results/'
    slash = "/"

trn_data, val_data, tst_data = preprocess_ptb_files(ptb_dir + 'ptb.train.txt',
                                                    ptb_dir + 'ptb.valid.txt',
                                                    ptb_dir + 'ptb.test.txt')  ## TODO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trn_dataset = create_dataset(trn_data, batch_size, sequence_length, device)
val_dataset = create_dataset(val_data, batch_size, sequence_length, device)
tst_dataset = create_dataset(tst_data, batch_size, sequence_length, device)

now = datetime.now()
time = now.strftime('%d_%m_%y_%H_%M')
checkpoints_dir_path = f'{checkpoints_dir_path}/{time}'
if not os.path.exists(checkpoints_dir_path):
    os.makedirs(f'{checkpoints_dir_path}/checkpoints')
    os.makedirs(f'{checkpoints_dir_path}/events')

events_dir = checkpoints_dir_path + slash + 'events' + slash
writer = SummaryWriter(events_dir)

mode = 'train'
variation_list = ['LSTM', 'LSTM-DO', 'GRU', 'GRU-DO']
## train all models with dropout, weight decay and batch normalization options
if mode == 'train':
    checkpoint_e_start = 0
    for variation in variation_list:

        if (checkpoint_e_start > 0):
            cpt_path = checkpoints_dir_path + f"/Zaremba-{variation}-{checkpoint_e_start}.pth"
        else:
            cpt_path = ""

        model = Zaremba(word_vec_size, word_vec_size, vocab_size, num_layers, variation)
        optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
        model.to(device)
        print(f"Starting train for {variation} variation")
        train(model, trn_dataset[:5], val_dataset[:5], tst_dataset[:5], batch_size, sequence_length, lr, lr_factor, lr_change_epoch,
              max_grad_norm, device, variation, optimizer, epoch_num, checkpoints_dir_path, writer)
