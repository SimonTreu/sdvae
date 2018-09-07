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
        self.lambda_kl = opt.lambda_kl
        self.input_size = opt.fine_size ** 2

        d_hidden = opt.d_hidden
        threshold = opt.threshold

        # hidden layer (shared by mu and log_var):
        hidden_layer = [nn.Conv2d(in_channels=1, out_channels=d_hidden,
                                  kernel_size=3, padding=1, stride=1),
                        nn.BatchNorm2d(d_hidden),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2)]

        # mu
        mu = [nn.Conv2d(in_channels=d_hidden, out_channels=self.nz,
                        kernel_size=4, padding=0, stride=1)]

        # log_var
        log_var = [nn.Conv2d(in_channels=d_hidden, out_channels=self.nz,
                             kernel_size=4, padding=0, stride=1)]

        self.mu = nn.Sequential(*hidden_layer, *mu)
        self.log_var = nn.Sequential(*hidden_layer, *log_var)

        self.decode = Decoder(nz=self.nz, no=self.no, threshold=threshold)

        if self.no > 0:
            self.encode_orog = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.no*2,
                                                       kernel_size=3, padding=1, stride=1),
                                             nn.BatchNorm2d(self.no*2),
                                             nn.ReLU(),
                                             nn.MaxPool2d(kernel_size=2),
                                             nn.Conv2d(in_channels=self.no*2, out_channels=self.no,
                                                       kernel_size=4, padding=0, stride=1)
                                             )
            self.encode_orog_2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.no * 2,
                                    kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(self.no * 2),
                          nn.ReLU(),
                          nn.Conv2d(in_channels=self.no * 2, out_channels=self.no,
                                    kernel_size=3, padding=1, stride=2)
                          )

    def forward(self, fine_pr, coarse_pr, orog):
        mu = self.mu(fine_pr)
        log_var = self.log_var(fine_pr)
        z = self.reparameterize(mu, log_var)
        if self.no > 0:
            o = self.encode_orog(orog)
            o2 = self.encode_orog_2(orog)
            return self.decode(z, coarse_pr, o, o2), mu.view(-1, self.nz), log_var.view(-1, self.nz)
        else:
            return self.decode(z, coarse_pr), mu.view(-1, self.nz), log_var.view(-1, self.nz)

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
        MSE = nn.functional.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * self.lambda_kl

        # cycle loss as mean squared error
        recon_average = get_average(recon_x.view(-1,64), cell_area.contiguous().view(-1, self.input_size))
        cycle_loss = torch.sum(torch.abs(coarse_pr.view(-1).sub(recon_average))) * self.lambda_cycle_l1

        return MSE, KLD, cycle_loss, MSE + KLD + cycle_loss


class Decoder(nn.Module):
    def __init__(self, nz, no, threshold, hidden_depth=None):
        super(Decoder, self).__init__()
        self.nz = nz
        self.no = no

        decoder_input_size = self.nz+self.no+1
        if hidden_depth is None:
            hidden_depth = decoder_input_size * 8

        # todo variable number of filters
        # todo skip connections
        # todo add in coarse pr. at several points

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(in_channels=decoder_input_size,
                                                       out_channels=hidden_depth,
                                                       kernel_size=4, stride=1, padding=0),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(in_channels=hidden_depth + 1 +self.no,
                                                       out_channels=1, kernel_size=4,
                                                       stride=2, padding=1),
                                    nn.Threshold(value=threshold, threshold=threshold)
                                    )

    def forward(self, z, coarse_pr, o=None, o2=None):
        if o is None:
            hidden_state = self.layer1(torch.cat((z, coarse_pr), 1))
        else:
            hidden_state = self.layer1(torch.cat((z, coarse_pr, o), 1))
        return self.layer2(torch.cat((hidden_state,
                                      coarse_pr.expand(-1, -1, hidden_state.shape[-2], hidden_state.shape[-1]),
                                      o2
                                      )
                                     , 1)
                           )
