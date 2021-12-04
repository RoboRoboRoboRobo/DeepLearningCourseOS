from model import Zaremba, Embedding ## move to notebook
from preprocess import preprocess_ptb_files, create_dataset
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from train import train

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
variation = 'LSTM'

"""" paths:
OFIR   C:\\Users\ofir-kr\PycharmProjects\DeepLearningCourseOS\EX2\\"
SHIR   /Users/shir.barzel/DeepLearningCourseOS/EX2/
"""
user = "Ofir"
if user == "Ofir":
    slash ="\\"
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

# now = datetime.now()
# time = now.strftime('%d_%m_%y_%H_%M')
# path_results_dir = f'{root_path}/ex2_301917670_302921366/results/{time}'
# if not os.path.exists(path_results_dir):
#   os.makedirs(f'{path_results_dir}/checkpoints')
#   os.makedirs(f'{path_results_dir}/events')

# events_dir = checkpoints_dir_path + slash + 'events' + slash
# writer = SummaryWriter(events_dir)

model = Zaremba(word_vec_size, word_vec_size, vocab_size, num_layers, variation)
embed = Embedding(vocab_size, word_vec_size)
optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)

model.to(device)
print("starting train")
train(model, trn_dataset, val_dataset, tst_dataset, batch_size, sequence_length, lr, lr_factor, lr_change_epoch,
      max_grad_norm, embed, device, variation, optimizer, epoch_num, checkpoints_dir_path, writer=0)