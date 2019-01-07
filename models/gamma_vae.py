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
        self.model = opt.model
        self.coarse_layer3 = not opt.no_coarse_layer3
        self.coarse_layer4 = not opt.no_coarse_layer4
        self.scale_factor = opt.scale_factor
        self.fine_size = opt.fine_size
        self.device = device


        # dimensions for batch_size=64, nf_encoder=16, fine_size=32, nz=10, orog=True, coarse_layer3 = True, coarse_layer4 = True
        # 64x5x32x32
        self.h_layer1 = self._down_conv(in_channels=1 + self.use_orog + self.coarse_layer4 * 3, out_channels=self.nf_encoder,
                                        kernel_size=3, padding=1, stride=1)
        # 64x20x16x16
        self.h_layer2 = self._down_conv(in_channels=self.nf_encoder + self.use_orog + self.coarse_layer3 * 3,
                                        out_channels=self.nf_encoder * 2,
                                        kernel_size=4, padding=0, stride=1)
        # 64x35x6x6
        self.h_layer3 = self._down_conv(in_channels=2 * self.nf_encoder + 3, out_channels=self.nf_encoder * 3,
                                        kernel_size=3, padding=1, stride=1)
        # 64x48x3x3

        # mu
        self.mu = nn.Sequential(nn.Linear(in_features=self.nf_encoder * 3 * 9, out_features=self.nz))
        # 64x10x1x1

        # log_var
        self.log_var = nn.Sequential(nn.Linear(in_features=self.nf_encoder * 3 * 9, out_features=self.nz))
        # 64x10x1x1

        self.decode = Decoder(opt, device)

    def forward(self, fine_pr, coarse_pr, orog, coarse_uas, coarse_vas):
        # layer 1
        upsample32 = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        coarse_pr_32 = upsample32(coarse_pr[:, :, 1:-1, 1:-1])
        coarse_uas_32 = upsample32(coarse_uas[:, :, 1:-1, 1:-1])
        coarse_vas_32 = upsample32(coarse_vas[:, :, 1:-1, 1:-1])
        input_layer_input = [fine_pr]
        if self.use_orog:
            input_layer_input.append(orog)
        if self.coarse_layer4:  # todo rename to coarse 32
            input_layer_input += [coarse_pr_32, coarse_uas_32, coarse_vas_32]
        h_layer1 = self.h_layer1(torch.cat(input_layer_input, 1))
        # layer 2
        upsample16 = torch.nn.Upsample(scale_factor=self.scale_factor // 2, mode='nearest')
        coarse_pr_16 = upsample16(coarse_pr[:, :, 1:-1, 1:-1])
        coarse_uas_16 = upsample16(coarse_uas[:, :, 1:-1, 1:-1])
        coarse_vas_16 = upsample16(coarse_vas[:, :, 1:-1, 1:-1])
        upscale16 = Upscale(size=self.fine_size, scale_factor=2, device=self.device)
        orog16 = upscale16.upscale(orog)
        layer2_input = [h_layer1]
        if self.use_orog:
            layer2_input.append(orog16)
        if self.coarse_layer3:
            layer2_input += [coarse_pr_16, coarse_uas_16, coarse_vas_16]
        h_layer2 = self.h_layer2(torch.cat(layer2_input, 1))
        # layer 3
        h_layer3 = self.h_layer3(torch.cat((h_layer2, coarse_pr, coarse_uas, coarse_vas), 1))
        # output layer
        mu = self.mu(h_layer3.view(h_layer3.shape[0], -1)).unsqueeze(-1).unsqueeze(-1)
        log_var = self.log_var(h_layer3.view(h_layer3.shape[0], -1)).unsqueeze(-1).unsqueeze(-1)
        # reparameterization
        z = self._reparameterize(mu, log_var)
        # decode
        recon_pr = self.decode(z=z, coarse_pr=coarse_pr,orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas)
        return recon_pr, mu.view(-1, self.nz), log_var.view(-1, self.nz)

    def loss_function(self, recon_x, x, mu, log_var, coarse_pr,):
        # negative log predictive density
        if self.model == 'gamma_vae':
            nlpd = self._neg_log_gamma_likelihood(x, recon_x['alpha'], recon_x['beta'], recon_x['p'])
        elif self.model == 'mse_vae':
            nlpd = nn.functional.mse_loss(recon_x, x, size_average=False)/x.shape[0]

        # Kullback-Leibler Divergence
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),1))
        loss = kld + nlpd

        # cycle loss
        # not added to the total loss
        if self.model == 'gamma_vae':
            coarse_recon = self.upscaler.upscale(recon_x['p'] * recon_x['alpha'] * recon_x['beta'])
        elif self.model == 'mse_vae':
            coarse_recon = self.upscaler.upscale(recon_x)
        cycle_loss = nn.functional.mse_loss(coarse_pr[:,:,1:-1,1:-1], coarse_recon, size_average=True)

        return nlpd, kld, cycle_loss, loss

    def _reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

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
        return result/x.shape[0]  # mean over batch size

    def _down_conv(self, in_channels, out_channels, kernel_size, padding, stride):
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
    def __init__(self, opt, device):
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
        self.model = opt.model
        self.device = device

        # dimensions for batch_size=64, nf_decoder=16, fine_size=32, nz=10
        # 64x10x1x1
        self.layer1 = self._up_conv(in_channels=self.nz, out_channels=nf_decoder, kernel_size=6, stride=1, padding=0)
        # 64x19x6x6
        self.layer2 = self._up_conv(in_channels=nf_decoder + 3, out_channels=nf_decoder * 2,
                                    kernel_size=3, stride=3, padding=1)
        # 64x36x16x16
        self.layer3 = self._up_conv(in_channels=nf_decoder * 2 + self.use_orog + self.coarse_layer3 * 3,
                                    out_channels=nf_decoder * 2, kernel_size=4,
                                    stride=2, padding=1)
        # 64x36x32x32
        # all padding
        self.layer4 = self._conv(in_channels=nf_decoder * 2 + self.use_orog + self.coarse_layer4 * 3, out_channels=nf_decoder * 2,
                                 kernel_size=3, stride=1, padding=1)
        # 64x32x32x32
        # layer 4 cannot be the output layer to enable a nonlinear relationship with topography

        # output parameters for mixed bernoulli-gamma distribution
        if self.model == 'gamma_vae':
            self.p_layer = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2, out_channels=1,
                                                        kernel_size=3, stride=1, padding=1),
                                         nn.Sigmoid())
            self.alpha_layer = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2, out_channels=1,
                                                        kernel_size=3, stride=1, padding=1),
                                             ExpModule())
            self.beta_layer = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2, out_channels=1,
                                                        kernel_size=3, stride=1, padding=1),
                                            ExpModule())

        # output for normal noise process (mse loss function)
        elif self.model == 'mse_vae':
            self.output_layer = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2, out_channels=1,
                                                        kernel_size=3, stride=1, padding=1),
                                              nn.ReLU())

    def forward(self, z, coarse_pr,
                coarse_uas, coarse_vas, orog):
        # layer 1
        hidden_state = self.layer1(z)
        # layer 2
        hidden_state2 = self.layer2(torch.cat((hidden_state, coarse_pr, coarse_uas, coarse_vas), 1))
        # layer 3
        upsample16 = torch.nn.Upsample(scale_factor=self.scale_factor // 2, mode='nearest')
        coarse_pr_16 = upsample16(coarse_pr[:, :, 1:-1, 1:-1])
        coarse_uas_16 = upsample16(coarse_uas[:, :, 1:-1, 1:-1])
        coarse_vas_16 = upsample16(coarse_vas[:, :, 1:-1, 1:-1])
        upscale16 = Upscale(size=self.fine_size, scale_factor=2, device=self.device)
        orog16 = upscale16.upscale(orog)
        layer3_input = [hidden_state2]
        if self.use_orog:
            layer3_input.append(orog16)
        if self.coarse_layer3:
            layer3_input += [coarse_pr_16, coarse_uas_16, coarse_vas_16]
        hidden_state3 = self.layer3(torch.cat(layer3_input,1))
        # layer 4
        upsample32 = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        coarse_pr_32 = upsample32(coarse_pr[:, :, 1:-1, 1:-1])
        coarse_uas_32 = upsample32(coarse_uas[:, :, 1:-1, 1:-1])
        coarse_vas_32 = upsample32(coarse_vas[:, :, 1:-1, 1:-1])
        layer4_input = [hidden_state3]
        if self.use_orog:
            layer4_input.append(orog)
        if self.coarse_layer4:
            layer4_input += [coarse_pr_32, coarse_uas_32, coarse_vas_32]
        hidden_state4 = self.layer4(torch.cat(layer4_input, 1))
        # output layer
        if self.model == 'gamma_vae':
            p = self.p_layer(hidden_state4)
            alpha = self.alpha_layer(hidden_state4)
            beta = self.beta_layer(hidden_state4)
            output = {'p': p, 'alpha': alpha, 'beta': beta}
        elif self.model == 'mse_vae':
            output = self.output_layer(hidden_state4)
        else:
            raise ValueError("model {} is not implemented".format(self.model))
        return output

    def _up_conv(self, in_channels, out_channels, kernel_size, stride, padding):
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

    def _conv(self, in_channels, out_channels, kernel_size, padding, stride):
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


class ExpModule(nn.Module):
    def __init__(self):
        super(ExpModule, self).__init__()

    def forward(self, x):
        return torch.exp(x)