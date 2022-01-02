import torch
from torchvision import datasets, transforms  # operations over images
from torchvision.transforms import Normalize  # operations over images
from torch import optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
from gan_models import Generator, Discriminator
from train_gan import train_gan

#Assign cuda GPU located at location '0' to a variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

variation = 'dcgan'

# algorithm parameters
batch_size = 64
z_dim = 100
input_image_size = (batch_size, 1, 32, 32)
num_of_classes = 10
dim_channels = 1
epoch_num = 15
lr = 1e-4
lr_factor = 1.3
lambda_val = 10.0
num_of_disc_iter = 5
lr_change_epoch = 100
max_grad_norm = 100

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
elif user == "Colab":
    assignment_path = "/content/DeepLearningCourseOS/EX3/"
    checkpoints_dir_path = '/content/DeepLearningCourseOS/EX3/results/'
    slash = "/"

# save fashion MNIST dataset to drive
mnist_version = "fashion"

if mnist_version == "digits":
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), Normalize(mean=0.5, std=0.5)])  ## (*) the mean and std of the train dataset were obtained using the code below
    mnist_train_data = datasets.MNIST(assignment_path, download=True, transform=transform, train=True)
    mnist_test_data = datasets.MNIST(assignment_path, download=True, transform=transform, train=False)
elif mnist_version == "fashion":
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), Normalize(mean=0.5, std=0.5)]) ## (*) the mean and std of the train dataset were obtained using the code below
    mnist_train_data = datasets.FashionMNIST(assignment_path, download=True, transform=transform, train=True)
    mnist_test_data = datasets.FashionMNIST(assignment_path, download=True, transform=transform, train=False)

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
data_loader_train = torch.utils.data.DataLoader(mnist_train_data, batch_size=batch_size, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(mnist_test_data, batch_size=batch_size, shuffle=False)

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
        cpt_path = checkpoints_dir_path + f"/Gulrajani-{checkpoint_e_start}.pth"
    else:
        cpt_path = ""

    betas = (0.5, 0.999)
    generator = Generator(dim_z=z_dim, image_size=(input_image_size[2], input_image_size[3]),
                          dim_channels=dim_channels, mode=variation)
    generator_optimizer = optim.Adam(params=generator.parameters(), lr=lr, betas=betas)
    generator.to(device)

    discriminator = Discriminator(dim_z=z_dim, image_size=(input_image_size[2], input_image_size[3]),
                          dim_channels=dim_channels, mode=variation)
    discriminator_optimizer = optim.Adam(params=generator.parameters(), lr=lr, betas=betas)
    discriminator.to(device)

    train_gan(generator, discriminator, data_loader_train, batch_size, lr, lambda_val,
              device, generator_optimizer, discriminator_optimizer, epoch_num, checkpoints_dir_path, writer,
              num_of_disc_iter, variation,
              latest_checkpoint_path="")

    #
    # else:
    #     checkpoint_path = "/Users/shir.barzel/DeepLearningCourseOS/EX3/results/29_12_21_21_21/Kingsma-10.pth"
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     first_epoch = checkpoint['epoch']
    #
    # number_of_labeled_samples_options = [100, 600, 1000, 3000]
    # print(f"checkpoint path: {checkpoint_path} for mnist {mnist_version}")
    # for number_of_labeled_samples in number_of_labeled_samples_options:
    #     classifier = train_classifier(model.encoder, data_loader_train, number_of_labeled_samples, num_of_classes)
    #     accuracy = evaluate_classifier(classifier, model.encoder, data_loader_test)
    #     print(f"For {number_of_labeled_samples} label samples, accuracy: {accuracy * 100}, error: {(1 - accuracy) * 100}")