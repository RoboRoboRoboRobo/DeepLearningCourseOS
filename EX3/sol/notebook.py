from VAE import VAE
from Classifier import Classifier
from train_classifier import train_classifier
from train_vae import train_vae
import torch
from torchvision import datasets, transforms  # operations over images
from torchvision.transforms import Normalize  # operations over images
from torch import optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

#Assign cuda GPU located at location '0' to a variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# algorithm parameters
batch_size = 16
input_size = (batch_size, 28, 28)
num_of_classes = 10
h_dim = 230
z_dim = 50
epoch_num = 25
lr = 1e-3

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
transform = transforms.Compose([transforms.ToTensor(), Normalize(mean=0.2860, std=0.3530), lambda x: x.reshape(-1)]) ## (*) the mean and std of the train dataset were obtained using the code below

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

now = datetime.now()
time = now.strftime('%d_%m_%y_%H_%M')
checkpoints_dir_path = f'{checkpoints_dir_path}/{time}'
if not os.path.exists(checkpoints_dir_path):
    os.makedirs(f'{checkpoints_dir_path}/checkpoints')
    os.makedirs(f'{checkpoints_dir_path}/events')

events_dir = checkpoints_dir_path + slash + 'events' + slash
writer = SummaryWriter(events_dir)

mode = 'train'
## train all models with dropout, weight decay and batch normalization options
if mode == 'train':
    checkpoint_e_start = 0
    if (checkpoint_e_start > 0):
        cpt_path = checkpoints_dir_path + f"/Kingsma-{checkpoint_e_start}.pth"
    else:
        cpt_path = ""

    model = VAE(input_size[1]**2, h_dim, z_dim)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.03)
    model.to(device)
    print(f"Starting train VAE")
    train_vae(model, data_loader_train, data_loader_test, batch_size, lr,
          device, optimizer, 1, checkpoints_dir_path, writer,
          latest_checkpoint_path="")

    number_of_labeled_samples = 3000
    train_classifier(model.encoder, data_loader_train, data_loader_test, batch_size, lr,
          device, optimizer, epoch_num, checkpoints_dir_path, writer, number_of_labeled_samples,
          latest_checkpoint_path="")