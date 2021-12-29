import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, dim_z, image_size, dim_channels, mode="wgan"): # mode can be dcgan
        super(Generator, self).__init__()
        self.dim_z = dim_z
        self.dim = dim
        self.image_size = image_size
        self.mode = mode

        self.linear = nn.Linear(dim_z, 7*7) ## TODO
        self.relu = nn.Relu()
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
        x = self.deconv1(x)
        x = self.relu(x)
        if self.mode == "wgan":
            x = self.bc_norm1(x)
        x = self.deconv2(x)
        x = self.relu(x)
        if self.mode == "wgan":
            x = self.bc_norm2(x)
        x = self.conv2d(x)
        if self.mode == "wgan":
            x = self.tanh(x)
        else:
            x = self.sigmoid(x)
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
        self.relu = nn.Relu()
        self.leaky_relu = nn.LeakyRelu()
        self.conv2 = nn.Conv2d(in_channels=dim_channels, out_channels=dim_channels, kernel_size=3, stride=2)
        self.sigmoid = nn.Sigmoid()

        # 2 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 2) * (img_size / 2 ^ 2)
        output_size = int(8 * dim_channels * (image_size[0] / 4) * (image_size[1] / 4))
        self.linear = nn.Linear(output_size, 1)

    def forward(self, input):
        x = self.conv1(input)
        if self.mode == 'dcgan':
            x = self.leaky_relu(x)
            x = self.bc_norm(x)
        else:
            x = self.relu(x)
        x = self.conv2(input)
        if self.mode == 'dcgan':
            x = self.leaky_relu(x)
            x = self.bc_norm(x)
        else:
            x = self.relu(x)
        x = self.linear(x)
        # x = self.sigmoid(x)







