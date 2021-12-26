from urllib import request

import torch
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append("../semi-supervised")

torch.manual_seed(1337)
np.random.seed(1337)

cuda = torch.cuda.is_available()
print("CUDA: {}".format(cuda))

def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)



if __name__ == "__main__":
    from itertools import repeat
    from torch.autograd import Variable
    import torch
    from torch import nn  ## nn model
    import torch.nn.functional as F  # activation functions
    from torch.autograd import Variable
    from torchvision import datasets, transforms  # operations over images
    from torchvision.transforms import Normalize  # operations over images
    from torch import optim
    from datetime import datetime
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    import os

    # Assign cuda GPU located at location '0' to a variable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # algorithm parameters
    batch_size = 16
    input_size = (batch_size, 28, 28)
    num_of_classes = 10
    h_dim = 230
    z_dim = 10
    epoch_num = 25
    lr = 0.001

    """" paths:
    OFIR   C:\\Users\ofir-kr\PycharmProjects\DeepLearningCourseOS\EX2\\"
    SHIR   /Users/shir.barzel/DeepLearningCourseOS/EX2/
    """
    user = "Shir"

    if user == "Ofir":
        slash = "\\"
        assignment_path = "C:\\Users\\ofir-kr\\PycharmProjects\\DeepLearningCourseOS\\EX3\PTB\\"
        checkpoints_dir_path = 'C:\\Users\ofir-kr\\PycharmProjects\\DeepLearningCourseOS\\EX3\\results\\'
    elif user == "Shir":
        assignment_path = "/Users/shir.barzel/DeepLearningCourseOS/EX3/"
        checkpoints_dir_path = '/Users/shir.barzel/DeepLearningCourseOS/EX3/results/'
        slash = "/"

    # convert data to tensor
    transform = transforms.Compose([transforms.ToTensor(), Normalize(mean=0.2860, std=0.3530), lambda x: x.reshape(
        -1)])  ## (*) the mean and std of the train dataset were obtained using the code below

    # save fashion MNIST dataset to drive
    fashion_mnist_train_data = datasets.FashionMNIST(assignment_path, download=True, transform=transform, train=True)
    fashion_mnist_test_data = datasets.FashionMNIST(assignment_path, download=True, transform=transform, train=False)

    '''
    "The values of the input pixels are normalized so that the background level (white) corresponds to a 
    value of -0.1 and the foreground (black) corresponds to 1.175 This makes the mean input roughly 0 and 
    the variance roughly 1 which accelerates learning" [LeCun, 98]
    '''

    # ## (*) computing the mean and std of the train dataset
    # fashion_mnist_train_data.data = torch.Tensor.float(fashion_mnist_train_data.data) / 255
    # fashion_mnist_test_data.data = torch.Tensor.float(fashion_mnist_test_data.data) / 255

    # # obtain mean and standard deviation of the training dataset (used in 'transform' above)
    # mean = torch.Tensor.float(fashion_mnist_train_data.data).mean()
    # std = torch.Tensor.float(fashion_mnist_train_data.data).std()

    # get datasets loaders
    data_loader_train = torch.utils.data.DataLoader(fashion_mnist_train_data, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(fashion_mnist_test_data, batch_size=batch_size, shuffle=False)

    # unlabelled = DataLoader(dset, batch_size=16, shuffle=True, sampler=SubsetRandomSampler(np.arange(len(dset)//3)))

    models = []

    from models import VariationalAutoencoder
    model = VariationalAutoencoder([input_size[1]**2, z_dim, [h_dim, h_dim]])
    model.decoder = nn.Sequential(
        nn.Linear(z_dim, h_dim),
        nn.Tanh(),
        nn.Linear(h_dim, h_dim),
        nn.Tanh(),
        nn.Linear(h_dim, h_dim),
        nn.Tanh(),
        nn.Linear(h_dim, input_size[1]**2),
        nn.Sigmoid(),
    )

    if cuda: model = model.cuda()

    beta = repeat(4.0)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)

    epochs = 251
    best = np.inf

    file = open(model.__class__.__name__ + ".log", 'w+')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, _ in data_loader_train:
            u = Variable(u.float())

            if cuda:
                u = u.cuda(device=0)

            reconstruction = model(u)

            likelihood = -binary_cross_entropy(reconstruction, u)
            elbo = likelihood - next(beta) * model.kl_divergence

            L = -torch.mean(elbo)

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += L.data

        m = len(data_loader_train)
        print(total_loss / m, sep="\t")

        if total_loss < best:
            best = total_loss
            torch.save(model, '{}.pt'.format(model.__class__.__name__))
