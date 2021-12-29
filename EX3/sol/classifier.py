import torch
from sklearn import svm
import numpy as np

## TODO Remove?
# numDataPoints = 1000
# data_dim = 5
# bs = 100
#
# # Create dummy data with class imbalance 9 to 1
# data = torch.FloatTensor(numDataPoints, data_dim)
# target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),
#                     np.ones(int(numDataPoints * 0.1), dtype=np.int32)))
#
# print('target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
#
# class_sample_count = np.array(
#     [len(np.where(target == t)[0]) for t in np.unique(target)])
# weight = 1. / class_sample_count
# samples_weight = np.array([weight[t] for t in target])
#
# samples_weight = torch.from_numpy(samples_weight)
# samples_weigth = samples_weight.double()
# sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
#
# target = torch.from_numpy(target).long()
# train_dataset = torch.utils.data.TensorDataset(data, target)

def train_classifier(encoder, trn_dataset, number_of_labeled_samples, num_of_classes):

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
    return svc


def evaluate_classifier(classifier, encoder, data_loader):
    x_list = []
    y_list = []
    for i, (x, y) in enumerate(data_loader):
        for i, (x_sample, y_sample) in enumerate(zip(x, y)):
            x_list.append(x_sample.detach().numpy())
            y_list.append(int(y_sample))

    z_list, _, _ = encoder(torch.Tensor(x_list))
    y_hat_list = classifier.predict(z_list.detach().numpy())

    accuracy = 1 - np.mean(np.array(y_hat_list) != np.array(y_list))
    return accuracy