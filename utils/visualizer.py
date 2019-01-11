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
        self.csv_epoch_name = os.path.join('checkpoints', opt.name, 'loss_epoch.csv')
        self.eval_csv = os.path.join('checkpoints', opt.name, 'eval.csv')
        self.fine_size = opt.fine_size
        util.mkdir(self.image_path)
        if opt.load_epoch < 0:
            with open(self.csv_name, "w") as log_csv:
                csv_writer = csv.writer(log_csv, delimiter= ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                title =['epoch', 'iters', 'mse', 'kl', 'cycle', 'total', 'iter_time', 'iter_data_time']
                csv_writer.writerow(title)
            with open(self.csv_epoch_name, "w") as csv_epoch_name:
                csv_writer = csv.writer(csv_epoch_name, delimiter= ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                title =['epoch', 'mse', 'kl', 'cycle', 'total', 'epoch_time']
                csv_writer.writerow(title)
            with open(self.eval_csv, "w") as eval_csv:
                csv_writer = csv.writer(eval_csv, delimiter= ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                title =['epoch', 'val_mse', 'train_mse', 'val_kld', 'train_kld', 'val_cycle_loss', 'train_cycle_loss',
                        'val_loss', 'train_loss', 'inf_losses']
                csv_writer.writerow(title)

    def plot(self, fine_pr, recon_pr, image_name):
        vmin = 0
        vmax = 6
        fig, axes = plt.subplots(2, self.n_images, sharex='col', sharey='row')
        rand_idx = np.random.randint(0, self.opt.batch_size, self.n_images)
        for i in range(self.n_images):
            axes[0, i].imshow(fine_pr[rand_idx[i]].view(48, 48).cpu().detach().numpy(), vmin=vmin, vmax=vmax,
                              cmap=plt.get_cmap('jet'))
            axes[1, i].imshow(recon_pr[rand_idx[i]].view(48, 48).cpu().detach().numpy(), vmin=vmin, vmax=vmax,
                              cmap=plt.get_cmap('jet'))

        axes[0, 0].set_title('Original Precipitation')
        axes[1, 0].set_title('Reconstructed Precipitation')
        fig.savefig(os.path.join(self.image_path, image_name))
        plt.close(fig)

    def print(self, epoch, batch_idx, mse, kld, cycle_loss, loss, iter_time, iter_data_time, load_time):
        print('Train Epoch: {:<3} [{:<6}/{} ({:<2.0f}%)]{:>10}MSE Loss: {:<10.5}KL Loss: {:<10.5f}cycle Loss {:<10.5f}'
              'Loss: {:<10.5f}Iteration Time: {:<10.4f}Data Loading Time: {:<10.4f}'.format(
               epoch, batch_idx * self.opt.batch_size, self.training_size,
               100. * batch_idx / self.n_batches,
               '',
               mse,
               kld,
               cycle_loss,
               loss,
               iter_time,
               iter_data_time))
        # csv title: 'epoch', 'iters', 'mse', 'kl', 'cycle', 'total', 'iter_time', 'iter_data_time'
        with open(self.csv_name, "a") as log_csv:
            csv_writer = csv.writer(log_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = [epoch, batch_idx,
                   mse,
                   kld,
                   cycle_loss,
                   loss,
                   iter_time,
                   iter_data_time]
            csv_writer.writerow(row)

    def print_epoch(self, epoch, epoch_mse, epoch_kld, epoch_cycle_loss, epoch_loss, epoch_time):
        print('-----------------------------------------------------------------------------------')
        print('====> Epoch: {}, average MSE: {:.2f}, average KL loss: {:.2f}, '
              'average cycle loss: {:.2f}, average loss: {:.2f}, calculation time = {:.2f}'.format(
            epoch,
            epoch_mse / self.training_size,
            epoch_kld / self.training_size,
            epoch_cycle_loss / self.training_size,
            epoch_loss / self.training_size,
            epoch_time))
        print('------------------------------------------------------------------------------------')

        with open(self.csv_epoch_name, "a") as log_csv:
            csv_writer = csv.writer(log_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = [epoch,
                   epoch_mse / self.training_size,
                   epoch_kld / self.training_size,
                   epoch_cycle_loss / self.training_size,
                   epoch_loss / self.training_size,
                   epoch_time]
            csv_writer.writerow(row)

    def print_eval(self, epoch, val_mse, val_kld, val_cycle_loss, val_loss, inf_losses,
                   train_mse, train_kld, train_cycle_loss, train_loss):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Train Epoch: {:<3}'
              'Val MSE: {:.1f} | Train MSE:  {:<10.1f}'
              'Val KL: {:.1f} | Train KL: {:<10.1f}'
              'Val cycle {:.3f} | Train cycle: {:<10.3f}'
              'Val Loss: {:.1f} | Train Loss: {:<10.1f} '
              ' Inf Losses: {}'.format(
               epoch,
               val_mse, train_mse,
               val_kld, train_kld,
               val_cycle_loss, train_cycle_loss,
               val_loss, train_loss,
               inf_losses))
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        with open(self.eval_csv, "a") as eval_csv:
            csv_writer = csv.writer(eval_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = [epoch, val_mse, train_mse, val_kld, train_kld, val_cycle_loss, train_cycle_loss,
                   val_loss, train_loss, inf_losses]
            csv_writer.writerow(row)