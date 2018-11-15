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
        self.lambda_mse = opt.lambda_mse
        self.input_size = opt.fine_size ** 2
        self.upscaler = Upscale(size=opt.fine_size, scale_factor=8, device=device)
        self.use_orog = not opt.no_orog

        nf_encoder = opt.nf_encoder
        threshold = opt.threshold

        # todo also enable removing orog
        self.h_layer1 = nn.Sequential(nn.Conv2d(in_channels=1+self.use_orog, out_channels=nf_encoder,
                                  kernel_size=3, padding=1, stride=1),
                        nn.BatchNorm2d(nf_encoder),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2))

        self.h_layer2 = nn.Sequential(nn.Conv2d(in_channels=nf_encoder, out_channels=nf_encoder*2,
                                  kernel_size=4, padding=0, stride=1),
                        nn.BatchNorm2d(nf_encoder*2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2))
        # todo 3 is uas, vas + coarse_pr and should be a variable
        self.h_layer3 = nn.Sequential(nn.Conv2d(in_channels=2*nf_encoder+3, out_channels=nf_encoder*3,
                                  kernel_size=3, padding=1, stride=1),
                        nn.BatchNorm2d(nf_encoder*3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2))

        # mu
        self.mu = nn.Sequential(nn.Conv2d(in_channels=3*nf_encoder, out_channels=self.nz,
                        kernel_size=3, padding=0, stride=1))

        # log_var
        self.log_var = nn.Sequential(nn.Conv2d(in_channels=3*nf_encoder, out_channels=self.nz,
                             kernel_size=3, padding=0, stride=1))

        self.decode = Decoder(opt)

        if self.no > 0:
            # todo fix this part
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
        if self.use_orog:
            h_layer1 = self.h_layer1(torch.cat((fine_pr, orog),1))
        else:
            h_layer1 = self.h_layer1(fine_pr)
        h_layer2 = self.h_layer2(h_layer1)
        h_layer3 = self.h_layer3(torch.cat((h_layer2, coarse_pr, coarse_uas, coarse_vas), 1))

        mu = self.mu(h_layer3)
        log_var = self.log_var(h_layer3)
        z = self.reparameterize(mu, log_var)

        # todo remove this if and else here
        if self.no > 0:
            o = self.encode_orog(orog)
            o2 = self.encode_orog_2(orog)
            return self.decode(z=z, coarse_pr=coarse_pr,
                               orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas,
                               o=o, o2=o2), mu.view(-1, self.nz), log_var.view(-1, self.nz)
        else:
            return self.decode(z=z, coarse_pr=coarse_pr,
                               orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas
                               ), mu.view(-1, self.nz), log_var.view(-1, self.nz)
        '''
        self.decode(z=self.reparameterize(torch.zeros_like(mu), torch.zeros_like(mu)), coarse_pr=coarse_pr,
                               orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas
                               ).detach().numpy()[0,0,:,:]
                               
        fine_pr.numpy()[0,0,:,:]
        
        mu.detach().numpy()[0,:,0,0]
        
import matplotlib.pyplot as plt
import numpy as np

vmin = -1.08
vmax = 20
fig, axes = plt.subplots(3, 5, sharex='col', sharey='row')
for i in range(5):
    axes[0, i].imshow(fine_pr.numpy()[0,0,:,:], vmin=vmin, vmax=vmax, cmap=plt.get_cmap('jet'))
    axes[1, i].imshow(self.decode(z=self.reparameterize(
        mu, log_var), coarse_pr=coarse_pr,orog=orog,
        coarse_uas=coarse_uas, coarse_vas=coarse_vas).detach().numpy()[0,0,:,:],
                      vmin=vmin, vmax=vmax, cmap=plt.get_cmap('jet'))
    axes[2, i].imshow(self.decode(z=self.reparameterize(
        torch.zeros_like(mu), torch.zeros_like(mu)), coarse_pr=coarse_pr,orog=orog,
        coarse_uas=coarse_uas, coarse_vas=coarse_vas).detach().numpy()[0,0,:,:],
                      vmin=vmin, vmax=vmax, cmap=plt.get_cmap('jet'))
axes[0, 0].set_title('Original Precipitation')
axes[1, 0].set_title('Reconstructed with encoded latent vector Precipitation')
axes[2, 0].set_title('Reconstructed Precipitation')
plt.show()
        '''

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, recon_x, x, mu, log_var, coarse_pr, lambda_kl):
        mse = nn.functional.mse_loss(recon_x, x, size_average=True)
        # see Appendix B from VAE paper:
        #Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # kld is devided with nz to normalize the kld values
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        coarse_recon = self.upscaler.upscale(recon_x)
        coarse_pr = coarse_pr[:,:,1:-1,1:-1]
        cycle_loss = nn.functional.mse_loss(coarse_pr, coarse_recon, size_average=True)
        loss = torch.zeros_like(mse)
        if self.lambda_mse != 0:
            mse *= self.lambda_mse
            loss += mse
        if self.lambda_kl != 0:
            kld *= lambda_kl
            loss += kld
        if self.lambda_cycle_l1 != 0:
            cycle_loss *= self.lambda_cycle_l1
            loss += cycle_loss
        return mse, kld, cycle_loss, loss


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.nz = opt.nz
        self.no = opt.no
        nf_decoder = opt.nf_decoder
        self.scale_factor = opt.scale_factor
        self.use_orog = not opt.no_orog


        # todo dropout

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.nz,
                                                       out_channels=nf_decoder,
                                                       kernel_size=6, stride=1, padding=0),
                                    nn.BatchNorm2d(nf_decoder),
                                    nn.ReLU())
        # todo put 3 (uas+vas+coarse_pr) to some variable
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(in_channels=nf_decoder + self.no + 3,
                                                       out_channels=nf_decoder * 2, kernel_size=3,
                                                       stride=3, padding=1),
                                    nn.BatchNorm2d(nf_decoder*2),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(in_channels=nf_decoder * 2 + 1,
                                                       out_channels=nf_decoder * 2, kernel_size=4,
                                                       stride=2, padding=1),
                                    nn.BatchNorm2d(nf_decoder*2),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2 + 1 + self.use_orog, out_channels=nf_decoder * 2,
                                              kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(nf_decoder*2),
                                    nn.ReLU())

        # layer 4 cannot be the output layer to enable a nonlinear relationship with topography

        self.output_layer = nn.Sequential(nn.Conv2d(in_channels=nf_decoder * 2, out_channels=1,
                                                    kernel_size=3, stride=1, padding=1),
                                          nn.Threshold(value=opt.threshold, threshold=opt.threshold))

    def forward(self, z, coarse_pr,
                coarse_uas, coarse_vas, orog,
                o=None):
        if o is None:
            hidden_state = self.layer1(z)
            hidden_state2 = self.layer2(torch.cat((hidden_state, coarse_pr, coarse_uas, coarse_vas), 1))
        else:
            hidden_state = self.layer1(z)
            hidden_state2 = self.layer2(torch.cat((hidden_state, o, coarse_pr, coarse_uas, coarse_vas), 1))

        upsample1 = torch.nn.Upsample(scale_factor=self.scale_factor//2, mode='nearest')
        coarse_pr_1 = upsample1(coarse_pr[:, :, 1:-1, 1:-1])

        hidden_state3 = self.layer3(torch.cat((hidden_state2, coarse_pr_1),1))

        upsample2 = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        coarse_pr_2 = upsample2(coarse_pr[:, :, 1:-1, 1:-1])

        if self.use_orog:
            hidden_state4 = self.layer4(torch.cat((hidden_state3, orog, coarse_pr_2), 1))
        else:
            hidden_state4 = self.layer4(torch.cat((hidden_state3, coarse_pr_2), 1))
        output = self.output_layer(hidden_state4)

        return output
