import torch.nn as nn
import torch
from utils.upscale import Upscale


class SDVAE(nn.Module):
    def __init__(self, opt, device):
        super(SDVAE, self).__init__()
        # variables
        self.nz = opt.nz
        self.input_size = opt.fine_size ** 2
        self.upscaler = Upscale(size=48, scale_factor=8, device=device)
        self.use_orog = not opt.no_orog
        self.no_dropout = opt.no_dropout
        self.nf_encoder = opt.nf_encoder
        self.model = opt.model
        self.scale_factor = opt.scale_factor
        self.fine_size = opt.fine_size
        self.device = device
        self.regression = opt.regression


        # dimensions for batch_size=64, nf_encoder=16, fine_size=32, nz=10, orog=True
        # 64x5x48x48
        self.h_layer1 = self._down_conv(in_channels=1 + self.use_orog + 4,
                                        out_channels=self.nf_encoder,
                                        kernel_size=4, padding=1, stride=2)
        # 64x20x24x24
        self.h_layer2 = self._down_conv(in_channels=self.nf_encoder + self.use_orog + 4,
                                        out_channels=self.nf_encoder * 2,
                                        kernel_size=4, padding=1, stride=2)
        # 64x20x12x12
        self.h_layer3 = self._down_conv(in_channels=self.nf_encoder*2 + self.use_orog + 4,
                                        out_channels=self.nf_encoder * 4,
                                        kernel_size=4, padding=1, stride=2)
        # 64x35x6x6
        self.h_layer4 = self._down_conv(in_channels=4 * self.nf_encoder + self.use_orog + 4
                                        , out_channels=self.nf_encoder * 8,
                                        kernel_size=4, padding=1, stride=2)
        # 64x48x3x3

        # mu
        self.mu = nn.Sequential(nn.Linear(in_features=self.nf_encoder * 8 * 9, out_features=self.nz))
        # 64x10x1x1

        # log_var
        self.log_var = nn.Sequential(nn.Linear(in_features=self.nf_encoder * 8 * 9, out_features=self.nz))
        # 64x10x1x1

        self.decode = Decoder(opt, device)

    def forward(self, fine_pr, coarse_pr, orog, coarse_uas, coarse_vas, coarse_psl):
        if not self.regression:
            # layer 1
            input_layer_input = [fine_pr]
            if self.use_orog:
                input_layer_input.append(orog)

            upsample48 = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
            input_layer_input += [upsample48(coarse_pr), upsample48(coarse_uas),
                                  upsample48(coarse_vas), upsample48(coarse_psl)]
            h_layer1 = self.h_layer1(torch.cat(input_layer_input, 1))

            # layer 2
            layer2_input = [h_layer1]
            if self.use_orog:
                upscale24 = Upscale(size=48, scale_factor=2, device=self.device)
                layer2_input.append(upscale24.upscale(orog))
            upsample24 = torch.nn.Upsample(scale_factor=4, mode='nearest')
            layer2_input += [upsample24(coarse_pr), upsample24(coarse_uas),
                             upsample24(coarse_vas), upsample24(coarse_psl)]
            h_layer2 = self.h_layer2(torch.cat(layer2_input, 1))

            # layer 3
            layer3_input = [h_layer2]
            if self.use_orog:
                upscale12 = Upscale(size=48, scale_factor=4, device=self.device)
                layer3_input.append(upscale12.upscale(orog))
            upsample12 = torch.nn.Upsample(scale_factor=2, mode='nearest')
            layer3_input += [upsample12(coarse_pr), upsample12(coarse_uas),
                             upsample12(coarse_vas), upsample12(coarse_psl)]
            h_layer3 = self.h_layer3(torch.cat(layer3_input, 1))

            # layer 4
            layer4_input = [h_layer3]
            if self.use_orog:
                upscale6 = Upscale(size=48, scale_factor=8, device=self.device)
                layer4_input.append(upscale6.upscale(orog))
            layer4_input += [coarse_pr, coarse_uas, coarse_vas, coarse_psl]
            h_layer4 = self.h_layer4(torch.cat(layer4_input, 1))

            # output layer
            mu = self.mu(h_layer4.view(h_layer4.shape[0], -1)).unsqueeze(-1).unsqueeze(-1)
            log_var = self.log_var(h_layer4.view(h_layer4.shape[0], -1)).unsqueeze(-1).unsqueeze(-1)
        else:
            mu = torch.zeros(fine_pr.shape[0], self.nz, 1, 1, dtype=torch.float, device=self.device)
            log_var = torch.zeros(fine_pr.shape[0], self.nz, 1, 1, dtype=torch.float, device=self.device)
        # reparameterization
        z = self._reparameterize(mu, log_var)

        # decode
        recon_pr = self.decode(z=z, coarse_pr=coarse_pr,orog=orog,
                               coarse_uas=coarse_uas, coarse_vas=coarse_vas, coarse_psl=coarse_psl)
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
        cycle_loss = nn.functional.mse_loss(coarse_pr, coarse_recon, size_average=True)

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
                                 nn.ReLU())
        else:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, padding=padding, stride=stride),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
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
        self.model = opt.model
        self.device = device

        # dimensions for batch_size=64, nf_decoder=16, fine_size=32, nz=10
        # 64x10x1x1
        self.layer1 = self._conv(in_channels=self.nz,
                                    out_channels=nf_decoder * 4,
                                    kernel_size=3, stride=1, padding=1)
        # 64x19x3x3
        self.layer2 = self._conv(in_channels=nf_decoder * 4 + self.use_orog + 4,
                                    out_channels=nf_decoder * 3,
                                    kernel_size=3, stride=1, padding=1)
        # 64x19x6x6
        self.layer3 = self._conv(in_channels=nf_decoder * 3 + self.use_orog + 4,
                                    out_channels=nf_decoder * 2,
                                    kernel_size=3, stride=1, padding=1)
        # 64x36x12x12
        self.layer4 = self._conv(in_channels=nf_decoder * 2 + self.use_orog + 4,
                                    out_channels=nf_decoder,
                                    kernel_size=3, stride=1, padding=1)
        # 64x36x24x24
        self.layer5 = self._conv(in_channels=nf_decoder + self.use_orog + 4,
                                    out_channels=nf_decoder,
                                    kernel_size=3,stride=1, padding=1)
        # 64x36x48x48
        # all padding
        self.layer6 = self._conv(in_channels=nf_decoder,
                                 out_channels=nf_decoder * 2,
                                 kernel_size=3, stride=1, padding=1)
        # 64x32x48x48
        # layer 6 cannot be the output layer to enable a nonlinear relationship with topography
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
                coarse_uas, coarse_vas, orog, coarse_psl):
        # layer 1
        hidden_state1 = self.layer1(torch.nn.Upsample(scale_factor=3, mode='nearest')(z))

        # layer 2 6x6
        layer2_input = [torch.nn.Upsample(scale_factor=2, mode='nearest')(hidden_state1)]
        if self.use_orog:
            upscale6 = Upscale(size=48, scale_factor=8, device=self.device)
            layer2_input.append(upscale6.upscale(orog))
        layer2_input += [coarse_pr, coarse_uas, coarse_vas, coarse_psl]
        hidden_state2 = self.layer2(torch.cat(layer2_input, 1))

        # layer 3 12x12
        layer3_input = [torch.nn.Upsample(scale_factor=2, mode='nearest')(hidden_state2)]
        if self.use_orog:
            upscale12 = Upscale(size=48, scale_factor=4, device=self.device)
            layer3_input.append(upscale12.upscale(orog))
        upsample12 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        layer3_input += [upsample12(coarse_pr), upsample12(coarse_uas),
                         upsample12(coarse_vas), upsample12(coarse_psl)]
        hidden_state3 = self.layer3(torch.cat(layer3_input, 1))

        # layer 4 24x24
        layer4_input = [torch.nn.Upsample(scale_factor=2, mode='nearest')(hidden_state3)]
        if self.use_orog:
            upscale24 = Upscale(size=48, scale_factor=2, device=self.device)
            layer4_input.append(upscale24.upscale(orog))
        upsample24 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        layer4_input += [upsample24(coarse_pr), upsample24(coarse_uas),
                         upsample24(coarse_vas), upsample24(coarse_psl)]
        hidden_state4 = self.layer4(torch.cat(layer4_input,1))

        # layer 5 48x48
        layer5_input = [torch.nn.Upsample(scale_factor=2, mode='nearest')(hidden_state4)]
        if self.use_orog:
            layer5_input.append(orog)
        upsample48 = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        layer5_input += [upsample48(coarse_pr), upsample48(coarse_uas),
                         upsample48(coarse_vas), upsample48(coarse_psl)]
        hidden_state5 = self.layer5(torch.cat(layer5_input,1))

        # layer 6
        hidden_state6 = self.layer6(hidden_state5)

        # output layer
        if self.model == 'gamma_vae':
            p = self.p_layer(hidden_state6)
            alpha = self.alpha_layer(hidden_state6)
            beta = self.beta_layer(hidden_state6)
            output = {'p': p, 'alpha': alpha, 'beta': beta}
        elif self.model == 'mse_vae':
            output = self.output_layer(hidden_state6)
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