import numpy as np

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
    num_batches = len(data) // batch_size
    data = data.view(num_batches * batch_size, 1)
    dataset = []
    for bat_ind in range(num_batches):
        x = []
        y = []
        for j in range(1, batch_size):
            x_seq = data[bat_ind*batch_size + seq_length * (j - 1):bat_ind*batch_size+seq_length * j ]
            y_seq = data[bat_ind*batch_size + seq_length * (j - 1) + 1:bat_ind*batch_size+seq_length * j + 1]
            x.append(x_seq)
            y.append(y_seq)
        dataset.append((x, y))

