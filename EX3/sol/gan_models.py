import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, dim_z, image_size, dim_channels, mode="wgan"): # mode can be dcgan
        super(Generator, self).__init__()
        self.dim_z = dim_z
        self.dim = dim_channels
        self.image_size = image_size

        self.mode = mode

        self.linear = nn.Linear(dim_z, 7*7) ## TODO
        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=dim_channels, kernel_size=3, stride=2)
        self.bc_norm1 = nn.BatchNorm2d(dim_channels)  # wgan
        self.deconv2 = nn.ConvTranspose2d(in_channels=dim_channels, out_channels=dim_channels, kernel_size=3, stride=2)
        self.bc_norm2 = nn.BatchNorm2d(dim_channels)  # wgan
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid() # wgan
        self.conv2d = nn.Conv2d(in_channels=dim_channels, out_channels=1, kernel_size=3, stride=1)

    def forward(self, input):
        x = self.linear(input)
        x = self.relu(x)
        mat_shape = int(x.shape[1] ** 0.5)
        x = x.view(-1, 1, mat_shape, mat_shape)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.bc_norm1(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.bc_norm2(x)
        x = self.conv2d(x)
        x = x[:, :, :-1, :-1] ### TODO
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dim_z, image_size, dim_channels, mode="wgan"): # mode can be dcgan
        super(Discriminator, self).__init__()
        self.dim_z = dim_z
        self.dim = dim_channels
        self.image_size = image_size
        self.mode = mode

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=dim_channels, kernel_size=3, stride=2)
        self.bc_norm = nn.BatchNorm2d(dim_channels)  # dcgan
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.conv2_dcgan = nn.Conv2d(in_channels=dim_channels, out_channels=dim_channels, kernel_size=3, stride=2)
        self.conv2_wgan = nn.Conv2d(in_channels=dim_channels, out_channels=dim_channels, kernel_size=3)
        self.mean_pool = nn.AvgPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # output_size = int(8 * dim_channels * (image_size[0] / 4) * (image_size[1] / 4))
        if mode == "wgan":
            output_size = 3200 # TODO
        elif mode == "dcgan":
            output_size = 4608  # TODO
        self.linear = nn.Linear(output_size, 1)

    def forward(self, input):
        if self.mode == 'dcgan':
            x = self.conv1(input)
            x = self.leaky_relu(x)
            x = self.bc_norm(x)
            x = self.conv2_dcgan(x)
            x = self.leaky_relu(x)
            x = self.bc_norm(x)
            x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
            x = self.linear(x)

            # x = self.tanh(x)

        elif self.mode == 'wgan':
            x = self.conv1(input)
            x = self.relu(x)
            x = self.conv2_wgan(x)
            x = self.relu(x)
            x = self.mean_pool(x)
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            x = self.linear(x)

        x = self.sigmoid(x)
        return x









