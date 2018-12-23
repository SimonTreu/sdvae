import torch.nn as nn
import torch
from utils.upscale import Upscale


class GammaVae(nn.Module):
    def __init__(self, opt, device):
        super(GammaVae, self).__init__()
        # variables
        self.nz = opt.nz
        self.input_size = opt.fine_size ** 2
        self.upscaler = Upscale(size=opt.fine_size, scale_factor=8, device=device)
        self.use_orog = not opt.no_orog
        self.no_dropout = opt.no_dropout
        self.nf_encoder = opt.nf_encoder

        self.h_layer1 = self.down_conv(in_channels=1 + self.use_orog, out_channels=self.nf_encoder,
                                       kernel_size=3, padding=1, stride=1)

        self.h_layer2 = self.down_conv(in_channels=self.nf_encoder, out_channels=self.nf_encoder * 2,
                                       kernel_size=4, padding=0, stride=1)

        self.h_layer3 = self.down_conv(in_channels=2 * self.nf_encoder + 3, out_channels=self.nf_encoder * 3,
                                                kernel_size=3, padding=1, stride=1)

        # mu
        self.mu = nn.Sequential(nn.Linear(in_features=self.nf_encoder * 3 * 9, out_features=self.nz))

        # log_var
        self.log_var = nn.Sequential(nn.Linear(in_features=self.nf_encoder * 3 * 9, out_features=self.nz))

        self.decode = Decoder(opt)

    def forward(self, fine_pr, coarse_pr, orog, coarse_uas, coarse_vas):
        if self.use_orog:
            h_layer1 = self.h_layer1(torch.cat((fine_pr, orog), 1))
        else:
            h_layer1 = self.h_layer1(fine_pr)
        h_layer2 = self.h_layer2(h_layer1)
        h_layer3 = self.h_layer3(torch.cat((h_layer2, coarse_pr, coarse_uas, coarse_vas), 1))

        mu = self.mu(h_layer3.view(h_layer3.shape[0], -1)).unsqueeze(-1).unsqueeze(-1)
        log_var = self.log_var(h_layer3.view(h_layer3.shape[0], -1)).unsqueeze(-1).unsqueeze(-1)
        z = self.reparameterize(mu, log_var)

        return (*self.decode(z=z, coarse_pr=coarse_pr,
                             orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas
                             )), mu.view(-1, self.nz), log_var.view(-1, self.nz)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, p, alpha, beta, x, mu, log_var, coarse_pr,):
        # negative log predictive density
        nlpd = self._neg_log_gamma_likelihood(x, alpha, beta, p)
        # Kullback-Leibler Divergence
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),1))
        loss = kld + nlpd

        # not added to the total loss
        coarse_recon = self.upscaler.upscale(p * alpha * beta)
        cycle_loss = nn.functional.mse_loss(coarse_pr[:,:,1:-1,1:-1], coarse_recon, size_average=True)

        return nlpd, kld, cycle_loss, loss

    def _neg_log_gamma_likelihood(self, x, alpha, beta, p):
        result = torch.zeros(1)
        if (x > 0).any():
            result = - torch.sum(torch.log(p[x > 0])
                        + (alpha[x > 0] - 1) * torch.log(x[x > 0])
                        - alpha[x > 0] * torch.log(beta[x > 0])
                        - x[x > 0] / beta[x > 0]
                        - torch.lgamma(alpha[x > 0])
                        )
        if (x == 0).any():
            result -= torch.sum(torch.log(1 - p[x == 0]) + 0 * alpha[x == 0] + 0 * beta[x == 0])
        return result/x.shape[0] # mean over batch size

    def down_conv(self, in_channels, out_channels, kernel_size, padding, stride):
        if self.no_dropout:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, padding=padding, stride=stride),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2))
        else:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, padding=padding, stride=stride),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Dropout())


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.nz = opt.nz
        nf_decoder = opt.nf_decoder
        self.scale_factor = opt.scale_factor
        self.use_orog = not opt.no_orog
        self.input_size = opt.fine_size ** 2
        self.fine_size = opt.fine_size
        self.no_dropout = opt.no_dropout
        self.coarse_layer3 = not opt.no_coarse_layer3
        self.coarse_layer4 = not opt.no_coarse_layer4

        self.layer1 = self.up_conv(in_channels=self.nz,out_channels=nf_decoder,kernel_size=6, stride=1, padding=0)
        self.layer2 = self.up_conv(in_channels=nf_decoder + 3,out_channels=nf_decoder * 2,
                                   kernel_size=3,stride=3, padding=1)
        self.layer3 = self.up_conv(in_channels=nf_decoder * 2 + self.coarse_layer3,
                                   out_channels=nf_decoder * 2, kernel_size=4,
                                   stride=2, padding=1)
        # all padding
        self.layer4 = self.conv(in_channels=nf_decoder * 2 + self.use_orog + self.coarse_layer4, out_channels=nf_decoder * 2,
                      kernel_size=3, stride=1, padding=2)

        # layer 4 cannot be the output layer to enable a nonlinear relationship with topography
        self.p_layer = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2, out_channels=1,
                                                    kernel_size=3, stride=1, padding=0),
                                     nn.Sigmoid())
        self.alpha_layer = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2, out_channels=1,
                                                    kernel_size=3, stride=1, padding=0),
                                         Exp_Module())
        self.beta_layer = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2, out_channels=1,
                                                    kernel_size=3, stride=1, padding=0),
                                        Exp_Module())

    def forward(self, z, coarse_pr,
                coarse_uas, coarse_vas, orog):
        hidden_state = self.layer1(z)
        hidden_state2 = self.layer2(torch.cat((hidden_state, coarse_pr, coarse_uas, coarse_vas), 1))

        upsample1 = torch.nn.Upsample(scale_factor=self.scale_factor // 2, mode='nearest')
        coarse_pr_1 = upsample1(coarse_pr[:, :, 1:-1, 1:-1])

        if self.coarse_layer3:
            hidden_state3 = self.layer3(torch.cat((hidden_state2, coarse_pr_1), 1))
        else:
            hidden_state3 = self.layer3(hidden_state2)

        upsample2 = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        coarse_pr_2 = upsample2(coarse_pr[:, :, 1:-1, 1:-1])

        layer4_input = [hidden_state3]
        if self.use_orog:
            layer4_input.append(orog)
        if self.coarse_layer4:
            layer4_input.append(coarse_pr_2)
        hidden_state4 = self.layer4(torch.cat(layer4_input, 1))
        p = self.p_layer(hidden_state4)
        alpha = self.alpha_layer(hidden_state4)
        beta = self.beta_layer(hidden_state4)

        return p, alpha, beta

    def up_conv(self, in_channels,out_channels, kernel_size, stride, padding):
        if self.no_dropout:
            return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())
        else:
            return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Dropout())

    def conv(self, in_channels, out_channels, kernel_size, padding, stride):
        if self.no_dropout:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, padding=padding, stride=stride),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())
        else:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, padding=padding, stride=stride),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Dropout())


class Exp_Module(nn.Module):
    def __init__(self):
        super(Exp_Module, self).__init__()

    def forward(self, x):
        return torch.exp(x)