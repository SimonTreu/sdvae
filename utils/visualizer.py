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
        with open(self.csv_name, "w") as log_csv:
            csv_writer = csv.writer(log_csv, delimiter= ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            title =['epoch', 'iters', 'mse', 'kl', 'cycle', 'total']
            csv_writer.writerow(title)

    def plot(self, fine_pr, recon_pr, image_name):
        vmin = self.opt.threshold
        vmax = 6
        fig, axes = plt.subplots(2, self.n_images, sharex='col', sharey='row')
        rand_idx = np.random.randint(0, self.opt.batch_size, self.n_images)
        for i in range(self.n_images):
            axes[0, i].imshow(fine_pr[rand_idx[i]].view(8, 8).detach().numpy(), vmin=vmin, vmax=vmax,
                              cmap=plt.get_cmap('jet'))
            axes[1, i].imshow(recon_pr[rand_idx[i]].view(8, 8).detach().numpy(), vmin=vmin, vmax=vmax,
                              cmap=plt.get_cmap('jet'))

        axes[0, 0].set_title('Original Precipitation')
        axes[1, 0].set_title('Reconstructed Precipitation')
        fig.savefig(os.path.join(self.image_path, image_name))
        plt.close(fig)

    def print(self, epoch, batch_idx, mse, kld, cycle_loss, loss):
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.7f}\tKL Loss: {:.7f}\tcycle Loss {:.7f}'
              '\tLoss: {:.7f}'.format(
            epoch, batch_idx * self.opt.batch_size, self.training_size,
            100. * batch_idx / self.n_batches,
            mse.item() / self.opt.batch_size,
            kld.item() / self.opt.batch_size,
            cycle_loss.item() / self.opt.batch_size,
            loss.item() / self.opt.batch_size))
        # csv title: 'epoch', 'iters', 'mse', 'kl', 'cycle', 'total' TODO add time and time_data
        with open(self.csv_name, "a") as log_csv:
            csv_writer = csv.writer(log_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = [epoch, batch_idx,
                   mse.item() / self.opt.batch_size,
                   kld.item() / self.opt.batch_size,
                   cycle_loss.item() / self.opt.batch_size,
                   loss.item() / self.opt.batch_size]
            csv_writer.writerow(row)