from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
import torch
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os


class Arg:
    # Todo do that with an arg_parser
    def __init__(self):
        self.dataroot = "data/wind"
        self.phase = "train"
        self.fine_size = 8
        self.batch_size = 124
        self.no_shuffle = False
        self.n_threads = 4
        # Number of hidden variables
        self.nz = 2
        self.gpu_ids = [-1]
        self.n_epochs = 20
        self.log_interval = 100
        self.plot_interval = 1e6
        self.lambda_cycle_l1 = 1000
        # load normalization values
        with Dataset(os.path.join(self.dataroot, "stats", "mean.nc4"), "r", format="NETCDF4") as rootgrp:
            mean = float(rootgrp.variables['pr'][:])

        with Dataset(os.path.join(self.dataroot, "stats", "std.nc4"), "r", format="NETCDF4") as rootgrp:
            std = float(rootgrp.variables['pr'][:])

        self.mean_std = {'mean': mean, 'std': std}
        self.threshold = (0-self.mean_std['mean'])/self.mean_std['std']
        self.name = 'vae_07_25'
        self.lr = 5e-3
        self.save_interval = 1


args = Arg()
device = torch.device("cuda" if args.gpu_ids[0] >= 0 else "cpu")

# create save dir
save_root = os.path.join('checkpoints', args.name)
if not os.path.isdir(save_root):
    os.makedirs(save_root)

# get the data
climate_data = ClimateDataset(opt=args)
climate_data_loader = DataLoader(climate_data,
                                 batch_size=args.batch_size,
                                 shuffle=not args.no_shuffle,
                                 num_workers=int(args.n_threads))

# load the model
edgan_model = Edgan(opt=args)
optimizer = torch.optim.Adam(edgan_model.parameters(), lr=args.lr)

for epoch in range(args.n_epochs):
    train_loss = 0
    edgan_model.train()
    for batch_idx, data in enumerate(climate_data_loader, 0):
        input_sample = data['input_sample'].to(device)
        average_value = data['average_value'].to(device)
        cell_area = data['cell_area'].to(device)

        optimizer.zero_grad()
        recon_x, mu, log_var = edgan_model(input_sample, average_value)
        bce, kld, cycle_loss, loss = edgan_model.loss_function(recon_x, input_sample, mu, log_var,
                                                               average_value, cell_area)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # todo add visualizer class
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBCE Loss: {:.7f}\tKL Loss: {:.7f}\tcycle Loss {:.7f}'
                  '\tLoss: {:.7f}'.format(
                    epoch, batch_idx * len(input_sample), len(climate_data_loader.dataset),
                    100. * batch_idx / len(climate_data_loader),
                    bce.item() / len(input_sample),
                    kld.item() / len(input_sample),
                    cycle_loss.item() / len(input_sample),
                    loss.item() / len(input_sample)))
            # todo make logging cluster ready
        if batch_idx % args.plot_interval == 0:
            vmin = args.threshold
            vmax = 2
            plt.imshow(input_sample[0].view(8, 8).detach().numpy(), vmin=vmin, vmax=vmax)
            plt.show()
            plt.imshow(recon_x[0].view(8, 8).detach().numpy(), vmin=vmin, vmax=vmax)
            plt.show()
            # todo make plotting cluster ready

    if epoch % args.save_interval == 0:
        save_name = "epoch_{}.pth".format(epoch)
        save_dir = os.path.join(save_root, save_name)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(edgan_model.module.cpu().state_dict(), save_dir)
            edgan_model.cuda(args.gpu_ids[0])
        else:
            torch.save(edgan_model.cpu().state_dict(), save_dir)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(climate_data_loader.dataset)))
