import torch.nn as nn
import torch
from utils.upscale import get_average


class Edgan(nn.Module):
    def __init__(self, opt):
        super(Edgan, self).__init__()
        # variables
        self.nz = opt.nz
        self.lambda_cycle_l1 = opt.lambda_cycle_l1
        self.input_size = opt.fine_size ** 2
        threshold = opt.threshold
        hidden_layer_size = opt.fine_size//2 ** 2

        # first layer (shared by mu and log_var):
        fc_layer_1 = nn.Linear(self.input_size, hidden_layer_size)
        relu_1 = nn.ReLU()
        # mu
        mu = nn.Linear(hidden_layer_size, self.nz)
        # log_var
        log_var = nn.Linear(hidden_layer_size, self.nz)
        self.mu = nn.Sequential(fc_layer_1, relu_1, mu)
        self.log_var = nn.Sequential(fc_layer_1, relu_1, log_var)

        # decoder
        self.decode = nn.Sequential(nn.Linear(self.nz+1, hidden_layer_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer_size, self.input_size),
                                    nn.Threshold(value=threshold, threshold=threshold)
                                    )

    def forward(self, x, average_value):
        x = x.view(-1, self.input_size)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(torch.cat((z,average_value),1)), mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # todo read if that can be defined somewhere else
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, log_var, average_value, cell_area):
        # todo change name of BCE
        BCE = nn.functional.mse_loss(recon_x, x.view(-1, self.input_size), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # cycle loss as mean squared error
        recon_average = get_average(recon_x, cell_area.contiguous().view(-1, self.input_size))
        cycle_loss = torch.mean(average_value.view(-1).sub(recon_average).pow(2)) * self.lambda_cycle_l1
        return BCE, KLD, cycle_loss, BCE + KLD + cycle_loss

