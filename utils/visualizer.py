import matplotlib.pyplot as plt
import os.path
import numpy as np
from utils import util
import csv


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
        # todo plot topography
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

    def print(self, epoch, batch_idx, mse, kld, cycle_loss, loss, iter_time, iter_data_time):
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
                   iter_data_time]
            csv_writer.writerow(row)