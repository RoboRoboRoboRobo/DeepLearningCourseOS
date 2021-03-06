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
    word2ind = {w: i for i, w in enumerate(vocab)}
    trn = [word2ind[w] for w in train_data]
    vld = [word2ind[w] for w in val_data]
    tst = [word2ind[w] for w in test_data]
    return np.array(trn).reshape(-1, 1), np.array(vld).reshape(-1, 1), np.array(tst).reshape(-1, 1)

def create_dataset(data, batch_size, seq_length, device):
    # Create list (samples, labels) of torch tensors of size (num_bat, batch_size, seq_length)
    data = torch.tensor(data, dtype=torch.long)
    if device.type == 'cuda':
      data = data.cuda()
    num_batches = len(data) // (batch_size * seq_length)
    data = data[:(data.size(0)//batch_size)*batch_size]
    data = data.view(batch_size, -1)
    dataset = []
    for bat_ind in range(0, num_batches - 1):
        x_batch = data[:, bat_ind * seq_length:bat_ind * seq_length + seq_length].transpose(1, 0)
        y_batch = data[:, bat_ind * seq_length + 1:bat_ind * seq_length + seq_length + 1].transpose(1, 0)
        dataset.append((x_batch, y_batch))
    return dataset

