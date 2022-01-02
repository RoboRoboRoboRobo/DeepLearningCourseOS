import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, dim_z, image_size, dim_channels, mode="wgan"): # mode can be dcgan
        super(Generator, self).__init__()
        self.dim_z = dim_z
        self.dim_channels = dim_channels
        self.image_size = image_size
        self.mode = mode

        self.deconv_linear = nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0)
        self.bc_norm_lin = nn.BatchNorm2d(num_features=1024)
        self.relu = nn.ReLU(True)

        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bc_norm1 = nn.BatchNorm2d(num_features=512)  # wgan

        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bc_norm2 = nn.BatchNorm2d(num_features=256)  # wgan

        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=dim_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        x = self.deconv_linear(input)
        x = self.bc_norm_lin(x)
        x = self.relu(x)
        x = self.deconv1(x)
        x = self.bc_norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.bc_norm2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dim_z, image_size, dim_channels, mode="wgan"): # mode can be dcgan
        super(Discriminator, self).__init__()
        self.dim_z = dim_z
        self.dim = dim_channels
        self.image_size = image_size
        self.mode = mode

        self.conv1 = nn.Conv2d(in_channels=dim_channels, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bc_norm = nn.InstanceNorm2d(256, affine=True)  # dcgan
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.relu = nn.ReLU()

        self.conv2_wgan = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bc_norm2_wgan = nn.InstanceNorm2d(512, affine=True)

        ## TODO
        self.conv2_dcgan = nn.Conv2d(in_channels=dim_channels, out_channels=dim_channels, kernel_size=3, stride=2)
        self.mean_pool = nn.AvgPool2d(kernel_size=2)

        self.conv3_wgan = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.bc_norm3_wgan = nn.InstanceNorm2d(1024, affine=True)

        self.conv4_wgan = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)

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
            x = self.bc_norm(x)
            x = self.leaky_relu(x)
            x = self.conv3_wgan(x)
            # x = self.tanh(x)

        elif self.mode == 'wgan':
            x = self.conv1(input)
            x = self.bc_norm(x)
            x = self.leaky_relu(x)

            x = self.conv2_wgan(x)
            x = self.bc_norm2_wgan(x)
            x = self.leaky_relu(x)

            x = self.conv3_wgan(x)
            x = self.bc_norm3_wgan(x)
            x = self.leaky_relu(x)

            x = self.conv4_wgan(x)
        return x









