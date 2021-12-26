import torch
from sklearn import svm
import numpy as np

numDataPoints = 1000
data_dim = 5
bs = 100

# Create dummy data with class imbalance 9 to 1
data = torch.FloatTensor(numDataPoints, data_dim)
target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),
                    np.ones(int(numDataPoints * 0.1), dtype=np.int32)))

print('target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))

class_sample_count = np.array(
    [len(np.where(target == t)[0]) for t in np.unique(target)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target])

samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

target = torch.from_numpy(target).long()
train_dataset = torch.utils.data.TensorDataset(data, target)


def train_classifier(encoder, trn_dataset, tst_dataset, batch_size, lr,
          device, optimizer, epoch_num, checkpoints_dir_path, writer, number_of_labeled_samples, num_of_classes,
          latest_checkpoint_path=""):

    x_train = []
    y_train = []

    max_samples_per_class = number_of_labeled_samples / num_of_classes
    dict = {}
    for i, (x, y) in enumerate(trn_dataset):
        if len(y_train) > number_of_labeled_samples:
            break
        for i, (x_sample, y_sample) in enumerate(zip(x,y)):
            if int(y_sample) not in dict:
                dict[int(y_sample)] = 1
                # z, _, _ = encoder(x)
                x_train.append(x_sample.detach().numpy())
                y_train.append(int(y_sample))
            elif dict[int(y_sample)] < max_samples_per_class:
                dict[int(y_sample)] += 1
                x_train.append(x_sample.detach().numpy())
                y_train.append(int(y_sample))

    z_train, _, _ = encoder(torch.Tensor(x_train))
    svc = svm.SVC(kernel='rbf', C=1).fit(z_train.detach().numpy(), np.array(y_train))

    ### evaluate on test!

    svc.predict(trn_dataset.dataset[0][0].detach().numpy())


        #
        #
        #     # move input and output to GPU
        #     if device.type == 'cuda':
        #         x = x.cuda()
        #
        #     # backward propagate
        #     z = encoder(x)
        #     y_hat = model(z)
        #
        #     L = tsvm_loss()
        #     # compute gradients
        #     L.backward()
        #
            # accumulate loss
            running_loss += L
        #
        #     optimizer.step()
        #     optimizer.zero_grad()
        #
            if i % 500 == 0 and i > 0:
                print(f"Train : Batch num: {i}/{len(trn_dataset)}, Loss: {running_loss / i}")
        #
        # model.eval()
        # acc_trn = evaluate_model(trn_dataset, model, device)
        # acc_tst = evaluate_model(tst_dataset, model, device)
        # # save checkpoint at the end of each epoch
        # curr_checkpoint_path = checkpoints_dir_path + f'/Kingsma-{e}.pth'
        # print(curr_checkpoint_path)
        # torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': running_loss / len(trn_dataset), },
        #             curr_checkpoint_path)
        #
        # writer.add_scalars(f'Kingsma', {
        #     'Train_Accuracy': float(acc_trn),
        #     'Test_Accuracy': float(acc_tst)
        # }, e)
        #
        # # print states at the end of each epoch
        # print(f"Epoch {e}:")
        # print(f"Training loss:     {running_loss / len(trn_dataset)}")
        # print(f"Training Accuracy: {acc_trn}")
        # print(f"Test Accuracy:     {acc_tst}")
        # print(f"-----------------------------------------------")

