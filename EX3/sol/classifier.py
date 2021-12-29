import torch
from sklearn import svm
import numpy as np

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