import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

def evaluateModel(dataset, model, embed):



    #### TODOD REMOVE

    dataset = dataset[:2]

    ####

    num_batches = len(dataset)
    batch_size = dataset[0][0].shape[0]
    seq_length = dataset[0][0].shape[1]
    vocab_size = embed.vocab_size
    tot_x = torch.zeros([num_batches * batch_size * seq_length, vocab_size])
    tot_y = torch.zeros(tot_x.shape[0])
    for i in range(0, num_batches, batch_size*seq_length):
        (x, y) = dataset[i]
        tot_x[i:i + batch_size*seq_length,:] = model(embed(x)).reshape(-1, vocab_size)
        tot_y[i:i + batch_size*seq_length] = y.reshape(-1)
    tot_x = embed(tot_x)
    output = model(tot_x)

def train(model, trn_dataset, val_dataset, tst_dataset, embed, device,
          variation, optimizer, epoch_num, checkpoints_dir_path, latest_checkpoint_path=""):

    train_acc_list = []
    test_acc_list = []
    epoch_list = []

    # checkpoints handling
    if latest_checkpoint_path == "":
        first_epoch = 0
    else:
        # load previous training checkpoint
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']

    # epoch iteration range
    epochs_range = range(first_epoch, epoch_num)
    print(f"epochs_range = {epochs_range}")

    perplexity_trn = evaluateModel(trn_dataset, model, embed)
    # epoch loop
    for e in epochs_range:
        running_loss = 0

        # batch loop
        for (x, y) in trn_dataset:
            # move input and output to GPU
            x = Variable(x).to(device)
            y = Variable(y).to(device)

            # clear gradients prior to new batch (since gradients are accumulated)
            optimizer.zero_grad()

            # obtain output probabilities (by feed forward images batch across the model)
            x = embed(x) ## seq_len x batch_size x word_vec_size (35x20x200)
            # y = embed(y) ## seq_len x batch_size x word_vec_size (35x20x200)

            x = model(x) ## tensor in the size of vocab_size over batch size (20 x 1 x 10,000)

            # obtain loss
            x = torch.exp(x)
            x = x.view(-1, vocab_size)
            y = y.reshape(batch_size * sequence_length)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(x, y)

            # compute gradients
            loss.backward()

            # update weights
            optimizer.step()

            # accumulate loss
            running_loss += loss.item()

        # deactivate dropout and batch normalization
        if ('DO' == variation):
            model.eval()
            # compute accuracies
        perplexity_trn = evaluateModel(trn_dataset, model)
        perplexity_val = evaluateModel(val_dataset, model)
        perplexity_tst = evaluateModel(tst_dataset, model)
        #
        # # accumulate accuracies
        # train_acc_list.append(acc_train)
        # test_acc_list.append(acc_test)
        # epoch_list.append(e)
        #
        # # save checkpint at the end of each epoch
        # curr_checkpoint_path = checkpoints_dir_path + f'/Lenet5-{variation}-{e}.pth'
        # print(curr_checkpoint_path)
        # torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': running_loss / len(data_loader_train), },
        #            curr_checkpoint_path)
        # writer.add_scalars(f'{variation}', {
        #     'Train Acc': acc_train,
        #     'Test Acc': acc_test
        # }, e)

        # save train and test accuracies figure after last epoch
        # if e == epoch_num - 1:
        #
        #     plt.figure()
        #     plt.plot(epoch_list, train_acc_list, label='Train Acc')
        #     plt.plot(epoch_list, test_acc_list, label='Test Acc')
        #     plt.xlabel("epochs")
        #     plt.ylabel("accuracy")
        #     var_type = "Original"
        #     if "DO" in variation:
        #         var_type = "Dropout"
        #     elif "WD" in variation:
        #         var_type = "Weight decay"
        #     elif "BN" in variation:
        #         var_type = "Batch Normalization"
        #     plt.title(f"Lenet5 - {var_type}")
        #     plt.legend()
        #     date_time = datetime.today().strftime('%d_%m_%Y_%H_%M_%S')
        #     str_figure = figures_dir_path + f'{variation}-{e}-{date_time}.png'
        #     plt.savefig(str_figure)
        #
        # # print states at the end of each epoch
        # print(f"Epoch {e}:")
        # print(f"Training loss:     {running_loss / len(data_loader_train)}")
        # print(f"Training accuracy: {acc_train}")
        # print(f"Test accuracy:     {acc_test}")
        # print(f"-----------------------------------------------")




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

train(model, trn_dataset, val_dataset, tst_dataset,
      embed, device, variation, optimizer, epoch_num, checkpoints_dir_path)
## in notebook

