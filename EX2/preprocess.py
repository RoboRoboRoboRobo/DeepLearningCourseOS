import numpy as np
import torch

def preprocess_ptb_files(train_path, val_path, test_path):
    with open(train_path) as f:
        file = f.read()
        train_data = file[1:].split(' ')
    with open(val_path) as f:
        file = f.read()
        val_data = file[1:].split(' ')
    with open(test_path) as f:
        file = f.read()
        test_data = file[1:].split(' ')
    vocab = sorted(set(train_data))
    trn_ind = np.asarray([vocab.index(word) for word in train_data])
    val_ind = np.asarray([vocab.index(word) for word in val_data])
    tst_ind = np.asarray([vocab.index(word) for word in test_data])
    return trn_ind, val_ind, tst_ind

def create_dataset(data, batch_size, seq_length):
    # Create list (samples, labels) of torch tensors of size (num_bat, batch_size, seq_length)
    data = torch.tensor(data)
    num_batches = len(data) // (batch_size * seq_length)
    data = data[:(data.size(0)//batch_size)*batch_size]
    data = data.view(batch_size, -1)
    dataset = []
    for bat_ind in range(0, num_batches - 1): ## TODO check end of batches
        x_batch = data[:, bat_ind * seq_length:bat_ind * seq_length + seq_length] ## TODO consider transpose
        y_batch = data[:, bat_ind * seq_length + 1:bat_ind * seq_length + seq_length + 1]
        dataset.append((x_batch, y_batch))
    return dataset

