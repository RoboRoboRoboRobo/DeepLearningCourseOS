import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import math

class Encoder(nn.Module):
    def __init__(self, dim_x, dim_hidden, dim_z):
        """ TODO REMOVE!!
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()
        self.linear_1 = nn.Linear(dim_x, dim_hidden)
        self.linear_2 = nn.Linear(dim_hidden, dim_hidden)
        self.gauss_params = GaussParamEst(dim_hidden, dim_z)


    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        z, mu_z, var_z = self.gauss_params(x)
        return z, mu_z, torch.log(var_z)

class GaussParamEst(nn.Module):
    def __init__(self, dim_hidden, dim_z):
        super(GaussParamEst, self).__init__()
        self.dim_z = dim_z
        self.mu = nn.Linear(dim_hidden, dim_z)
        self.var = nn.Linear(dim_hidden, dim_z)

    def forward(self, x):
        mu_z = self.mu(x)
        var_z = F.softplus(self.var(x))

        eps = Variable(torch.randn(mu_z.size()), requires_grad=False)
        if mu_z.is_cuda:
            eps.cuda()

        z = mu_z + var_z * eps

        return z, mu_z, var_z

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()
        self.linear_1 = nn.Linear(z_dim, h_dim)
        self.linear_2 = nn.Linear(h_dim, h_dim)
        self.linear_3 = nn.Linear(h_dim, x_dim)

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return self.output_activation(self.linear_3(x))


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VAE, self).__init__()

        self.z_dim = z_dim

        self.encoder = Encoder(x_dim, h_dim, z_dim)
        self.decoder = Decoder(x_dim, h_dim, z_dim)
        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(-0.08, 0.08)
                if m.bias is not None:
                    m.bias.data.zero_()

    def log_gaussian(self, w, mu, log_var):
        log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (w - mu) ** 2 / (2 * torch.exp(torch.Tensor(log_var)))
        return torch.sum(log_pdf, dim=-1)

    def _kld(self, z, q_param):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = - E[log (p(z) / q(z))]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        mu, log_var = q_param

        qz = self.log_gaussian(z, mu, log_var)
        pz = self.log_gaussian(z, torch.zeros(z.shape), torch.zeros(z.shape))

        kl = qz - pz

        return kl


    def forward(self, x):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, mu_z, log_var_z = self.encoder(x)
        # KLD for loss
        self.kl_divergence = self._kld(z, (mu_z, log_var_z))

        x_hat = self.decoder(z)

        return x_hat
