import torch.nn as nn
import torch
from utils.upscale import Upscale


class Edgan(nn.Module):
    def __init__(self, opt, device):
        super(Edgan, self).__init__()
        # variables
        self.nz = opt.nz
        self.no = opt.no # size of encoded orography
        self.lambda_cycle_l1 = opt.lambda_cycle_l1
        self.lambda_kl = opt.lambda_kl
        self.input_size = opt.fine_size ** 2
        self.upscaler = Upscale(size=opt.fine_size, scale_factor=8, device=device)

        d_hidden = opt.d_hidden
        threshold = opt.threshold

        # todo also enable removing orog
        # todo in_channels should be a variable
        self.h_layer1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=d_hidden,
                                  kernel_size=3, padding=1, stride=1),
                        nn.BatchNorm2d(d_hidden),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2))

        self.h_layer2 = nn.Sequential(nn.Conv2d(in_channels=d_hidden, out_channels=d_hidden*2,
                                  kernel_size=4, padding=0, stride=1),
                        nn.BatchNorm2d(d_hidden*2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2))
        # todo 3 is uas, vas + coarse_pr and should be a variable
        self.h_layer3 = nn.Sequential(nn.Conv2d(in_channels=2*d_hidden+3, out_channels=d_hidden*3,
                                  kernel_size=3, padding=1, stride=1),
                        nn.BatchNorm2d(d_hidden*3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2))

        # mu
        self.mu = nn.Sequential(nn.Conv2d(in_channels=3*d_hidden, out_channels=self.nz,
                        kernel_size=3, padding=0, stride=1))

        # log_var
        self.log_var = nn.Sequential(nn.Conv2d(in_channels=3*d_hidden, out_channels=self.nz,
                             kernel_size=3, padding=0, stride=1))

        self.decode = Decoder(nz=self.nz, no=self.no, threshold=threshold,
                              hidden_depth=opt.decoder_hidden_depth)

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

    def forward(self, fine_pr, coarse_pr, orog, coarse_uas, coarse_vas):
        h_layer1 = self.h_layer1(torch.cat((fine_pr, orog),1)) # shape = n_batch,n_channels, 6,6
        h_layer2 = self.h_layer2(h_layer1) # shape = n_batch,n_channels, 6,6
        h_layer3 = self.h_layer3(torch.cat((h_layer2, coarse_pr, coarse_uas, coarse_vas), 1))

        mu = self.mu(h_layer3)
        log_var = self.log_var(h_layer3)
        z = self.reparameterize(mu, log_var)

        # todo remove this if and else here
        if self.no > 0:
            o = self.encode_orog(orog)
            o2 = self.encode_orog_2(orog)
            # todo fix decoder
            return self.decode(z=z, coarse_pr=coarse_pr,
                               orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas,
                               o=o, o2=o2), mu.view(-1, self.nz), log_var.view(-1, self.nz)
        else:
            return self.decode(z=z, coarse_pr=coarse_pr,
                               orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas
                               ), mu.view(-1, self.nz), log_var.view(-1, self.nz)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def get_picture(self, coarse_precipitation,
                    coarse_ul, coarse_u, coarse_ur,
                    coarse_l, coarse_r,
                    coarse_bl, coarse_b, coarse_br,
                    orog,
                    latent=None):
        if latent is None:
            latent = torch.randn(coarse_precipitation.shape[0], self.nz, 1, 1)
        if self.no > 0:
            o = self.encode_orog(orog)
            o2 = self.encode_orog_2(orog)
            x_decoded = self.decode(latent, coarse_precipitation,
                                    coarse_ul, coarse_u, coarse_ur,
                                    coarse_l, coarse_r,
                                    coarse_bl, coarse_b, coarse_br,
                                    orog=orog,
                                    o=o, o2=o2)
        else:
            x_decoded = self.decode(latent, coarse_precipitation,
                                    coarse_ul, coarse_u, coarse_ur,
                                    coarse_l, coarse_r,
                                    coarse_bl, coarse_b, coarse_br,
                                    orog=orog
                                    )
        if coarse_precipitation.shape[0] == 1:
            return x_decoded.detach().view(8, 8)
        else:
            return x_decoded.detach()

    def loss_function(self, recon_x, x, mu, log_var, coarse_pr):
        mse = nn.functional.mse_loss(recon_x, x, size_average=False)
        # see Appendix B from VAE paper:
        #Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * self.lambda_kl
        # cycle loss as mean squared error
        coarse_recon = self.upscaler.upscale(recon_x)
        coarse_pr = coarse_pr[:,:,1:-1,1:-1]
        cycle_loss = torch.sum(torch.abs(coarse_pr.contiguous().view(-1).sub(coarse_recon.view(-1)))) * self.lambda_cycle_l1

        return mse, kld, cycle_loss, mse + kld + cycle_loss


class Decoder(nn.Module):
    def __init__(self, nz, no, threshold, hidden_depth):
        super(Decoder, self).__init__()
        self.nz = nz
        self.no = no
        self.hidden_depth = hidden_depth


        # todo skip connections
        # todo dropout or BatchNorm

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.nz,
                                                       out_channels=hidden_depth,
                                                       kernel_size=6, stride=1, padding=0),
                                    nn.ReLU())
        # todo put 3 (uas+vas+coarse_pr) to some variable
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(in_channels=hidden_depth+self.no+3,
                                                       out_channels=hidden_depth * 2, kernel_size=5,
                                                       stride=2, padding=0),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(in_channels=hidden_depth * 2,
                                                       out_channels=hidden_depth * 2, kernel_size=4,
                                                       stride=2, padding=0),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=hidden_depth * 2+1, out_channels=hidden_depth * 4,
                                              kernel_size=3, stride=1, padding=1),
                                    nn.ReLU())

        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=hidden_depth * 4, out_channels=1,
                                              kernel_size=3, stride=1, padding=1),
                                    nn.Threshold(value=threshold, threshold=threshold))

    def forward(self, z, coarse_pr,
                coarse_uas, coarse_vas, orog,
                o=None):
        if o is None:
            hidden_state = self.layer1(z)
            hidden_state2 = self.layer2(torch.cat((hidden_state, coarse_pr, coarse_uas, coarse_vas), 1))
        else:
            hidden_state = self.layer1(z)
            hidden_state2 = self.layer2(torch.cat((hidden_state, o, coarse_pr, coarse_uas, coarse_vas), 1))
        hidden_state3 = self.layer3(hidden_state2)
        # todo maybe add in the boundary conditions again at this point
        hidden_state4 = self.layer4(torch.cat((hidden_state3, orog), 1))
        hidden_state5 = self.layer5(hidden_state4)

        return hidden_state5
