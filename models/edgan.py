import torch.nn as nn
import torch
from utils.upscale import get_average


class Edgan(nn.Module):
    def __init__(self, opt):
        super(Edgan, self).__init__()
        # variables
        self.nz = opt.nz
        self.no = opt.no # size of encoded orography
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

        decoder_input_size = self.nz+self.no+1
        self.decode = nn.Sequential(nn.ConvTranspose2d(in_channels=decoder_input_size,
                                                       out_channels=decoder_input_size * 8,
                                                       kernel_size=4, stride=1, padding=0),
                                    nn.BatchNorm2d(decoder_input_size * 8),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=decoder_input_size*8,
                                                       out_channels=1, kernel_size=4,
                                                       stride=2, padding=1),
                                    nn.BatchNorm2d(1),
                                    nn.Threshold(value=threshold, threshold=threshold)
                                    )
        if self.no > 0:
            self.encode_orog = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.no//2,
                                                       kernel_size=3, padding=1, stride=1),
                                             nn.ReLU(),
                                             nn.MaxPool2d(kernel_size=2),
                                             nn.Conv2d(in_channels=self.no//2, out_channels=self.no,
                                                       kernel_size=3, padding=1, stride=1),
                                             nn.ReLU(),
                                             nn.MaxPool2d(kernel_size=4),
                                             )

    def forward(self, fine_pr, coarse_pr, orog):
        fine_pr = fine_pr.view(-1, self.input_size)
        mu = self.mu(fine_pr)
        log_var = self.log_var(fine_pr)
        z = self.reparameterize(mu, log_var)
        if self.no > 0:
            orog.unsqueeze_(1)  # bring into shape (N, n_ch, W, H)
            o = self.encode_orog(orog)
            return self.decode(torch.cat((z, coarse_pr, o.view(-1,self.no)), 1).unsqueeze(-1).unsqueeze(-1)), mu, log_var
        else:
            return self.decode(torch.cat((z, coarse_pr), 1)), mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # todo read if that can be defined somewhere else
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, log_var, coarse_pr, cell_area):
        # todo change name of BCE
        BCE = nn.functional.mse_loss(recon_x, x.unsqueeze(1), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # cycle loss as mean squared error
        recon_average = get_average(recon_x.view(-1,64), cell_area.contiguous().view(-1, self.input_size))
        cycle_loss = torch.mean(coarse_pr.view(-1).sub(recon_average).pow(2)) * self.lambda_cycle_l1
        return BCE, KLD, cycle_loss, BCE + KLD + cycle_loss

