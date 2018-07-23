import torch.nn as nn
import torch


class Edgan(nn.Module):
    def __init__(self, opt):
        super(Edgan, self).__init__()
        # variables
        self.nz = opt.nz
        self.input_size = self.fine_size ** 2
        hidden_layer_size = self.fine_size//2 ** 2

        # first layer (shared by mu and log_var):
        fc_layer_1 = nn.Linear(self.input_size, hidden_layer_size)
        relu_1 = nn.ReLU()
        # mu
        mu = nn.Linear(hidden_layer_size, self.nz)
        # log_var
        log_var = nn.Linear(hidden_layer_size, self.nz)
        self.mu = nn.Sequential(fc_layer_1, relu_1, mu)
        self.log_var = nn.Sequential(fc_layer_1, relu_1, log_var)

        # todo add in the coarse scale pr value as input for the decoder
        # decoder
        self.decode = nn.Sequential(nn.Linear(self.nz, hidden_layer_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer_size, self.input_size),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # todo read if that can be defined somewhere else
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, self.input_size), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE, KLD, BCE + KLD

