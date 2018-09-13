import matplotlib.pyplot as plt
import os.path
import numpy as np
from utils import util
import csv
import torch
from scipy.stats import norm
from matplotlib.widgets import Slider, Button



class Visualizer:
    def __init__(self, opt, n_images, training_size, n_batches):
        self.opt = opt
        self.image_path = os.path.join('checkpoints', opt.name, 'images')
        self.n_images = n_images
        self.training_size = training_size
        self.n_batches = n_batches
        self.csv_name = os.path.join('checkpoints', opt.name, 'loss.csv')
        util.mkdir(self.image_path)
        if opt.load_epoch < 0:
            with open(self.csv_name, "w") as log_csv:
                csv_writer = csv.writer(log_csv, delimiter= ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                title =['epoch', 'iters', 'mse', 'kl', 'cycle', 'total', 'iter_time', 'iter_data_time']
                csv_writer.writerow(title)

    def plot(self, fine_pr, recon_pr, image_name):
        vmin = self.opt.threshold
        vmax = 6
        fig, axes = plt.subplots(2, self.n_images, sharex='col', sharey='row')
        rand_idx = np.random.randint(0, self.opt.batch_size, self.n_images)
        for i in range(self.n_images):
            axes[0, i].imshow(fine_pr[rand_idx[i]].view(8, 8).cpu().detach().numpy(), vmin=vmin, vmax=vmax,
                              cmap=plt.get_cmap('jet'))
            axes[1, i].imshow(recon_pr[rand_idx[i]].view(8, 8).cpu().detach().numpy(), vmin=vmin, vmax=vmax,
                              cmap=plt.get_cmap('jet'))

        axes[0, 0].set_title('Original Precipitation')
        axes[1, 0].set_title('Reconstructed Precipitation')
        fig.savefig(os.path.join(self.image_path, image_name))
        plt.close(fig)

    def print(self, epoch, batch_idx, mse, kld, cycle_loss, loss, iter_time, iter_data_time, load_time):
        print('Train Epoch: {:<3} [{:<6}/{} ({:<2.0f}%)]{:>10}MSE Loss: {:<10.2f}KL Loss: {:<10.2f}cycle Loss {:<10.2f}'
              'Loss: {:<10.2f}Iteration Time: {:<10.4f}Data Loading Time: {:<10.4f}'.format(
               epoch, batch_idx * self.opt.batch_size, self.training_size,
               100. * batch_idx / self.n_batches,
               '',
               mse.item() / self.opt.batch_size,
               kld.item() / self.opt.batch_size,
               cycle_loss.item() / self.opt.batch_size,
               loss.item() / self.opt.batch_size,
               iter_time,
               iter_data_time))
        # csv title: 'epoch', 'iters', 'mse', 'kl', 'cycle', 'total', 'iter_time', 'iter_data_time'
        with open(self.csv_name, "a") as log_csv:
            csv_writer = csv.writer(log_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = [epoch, batch_idx,
                   mse.item() / self.opt.batch_size,
                   kld.item() / self.opt.batch_size,
                   cycle_loss.item() / self.opt.batch_size,
                   loss.item() / self.opt.batch_size,
                   iter_time,
                   iter_data_time,
                   load_time]
            csv_writer.writerow(row)


class ValidationViz:
    def __init__(self, opt):
        self.nz = opt.nz
        self.no = opt.no
        self.threshold = opt.threshold
        self.vmin = opt.threshold
        self.vmax = 7

    def plot_latent_walker(self, edgan_model, climate_data):
        fig2, ax = plt.subplots(figsize=(50,50))
        offset = 0.025 * (self.nz + 9)
        plt.subplots_adjust(bottom=0.15 + offset)

        # initial values
        z = torch.ones(1, self.nz, 1, 1) * 0.5
        coarse_pr = torch.ones(1, 1, 1, 1) * 0.5

        r = np.random.randint(len(climate_data))
        self.orog = climate_data[r]['orog'].unsqueeze(0)

        im = edgan_model.get_picture(latent=torch.Tensor(norm.ppf(z)), coarse_precipitation=coarse_pr,
                                     coarse_ul=coarse_pr, coarse_u=coarse_pr, coarse_ur=coarse_pr,
                                     coarse_l=coarse_pr, coarse_r=coarse_pr,
                                     coarse_bl=coarse_pr, coarse_b=coarse_pr,coarse_br=coarse_pr,
                                     orog=self.orog)
        img_in_plot = plt.imshow(im.detach().numpy(), origin='lower',
                                 cmap=plt.get_cmap('jet'), vmin=self.vmin, vmax=self.vmax)
        self.orog_in_plot = plt.contour(self.orog.view(8, 8))
        # position of the slider
        z_axis = [plt.axes([0.25, 0.05 + i_offset, 0.65, 0.03]) for i_offset in np.arange(offset, 0.0, -0.025)]
        z_sliders = [Slider(z_axis[i], 'Z {}'.format(i), 0, 1, valinit=z[0, i, 0, 0].item()) for i in range(self.nz)]
        z_sliders.append(Slider(z_axis[self.nz], 'Coarse Pr', self.threshold, 10, valinit=coarse_pr))
        z_sliders += [Slider(z_axis[i], ['ul', 'u', 'ur', 'l', 'r', 'bl', 'b', 'br'][i-self.nz-1],
                                 self.threshold, 10, valinit=coarse_pr) for i in range(self.nz+1, self.nz+9)]


        def update(val):
            for i in range(self.nz):
                z[0, i, 0, 0] = z_sliders[i].val
            coarse_pr[0, 0, 0, 0] = z_sliders[self.nz].val
            im = edgan_model.get_picture(latent=torch.Tensor(norm.ppf(z)), coarse_precipitation=coarse_pr,
                                         coarse_ul=torch.ones(1, 1, 1, 1) * z_sliders[self.nz+1].val,
                                         coarse_u=torch.ones(1, 1, 1, 1) * z_sliders[self.nz+2].val,
                                         coarse_ur=torch.ones(1, 1, 1, 1) * z_sliders[self.nz+3].val,
                                         coarse_l=torch.ones(1, 1, 1, 1) * z_sliders[self.nz+4].val,
                                         coarse_r=torch.ones(1, 1, 1, 1) * z_sliders[self.nz+5].val,
                                         coarse_bl=torch.ones(1, 1, 1, 1) * z_sliders[self.nz+6].val,
                                         coarse_b=torch.ones(1, 1, 1, 1) * z_sliders[self.nz+7].val,
                                         coarse_br=torch.ones(1, 1, 1, 1) * z_sliders[self.nz+8].val,
                                         orog=self.orog)
            img_in_plot.set_data(im.detach().numpy())
            fig2.canvas.draw_idle()
            fig2.suptitle('{}, {}'.format(coarse_pr.item(), torch.mean(im).item()))
            plt.draw()

        def update_orog(val):
            if self.no > 0:
                r = np.random.randint(len(climate_data))
                self.orog = climate_data[r]['orog'].unsqueeze(0)
            update(val)
            for coll in self.orog_in_plot.collections:
                coll.remove()
            self.orog_in_plot = ax.contour(self.orog.view(8, 8))
            plt.draw()

        for z_slider in z_sliders:
            z_slider.on_changed(update)

        ax_button = plt.axes([0.81, 0.0, 0.1, 0.075])
        b_orog = Button(ax_button, 'Orog')
        b_orog.on_clicked(update_orog)

        fig2.show()
        plt.show()