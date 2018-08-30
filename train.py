from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
import torch
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
from options.base_options import BaseOptions

opt = BaseOptions().parse()
device = torch.device("cuda" if len(opt.gpu_ids) > 0 else "cpu")

# create save dir
save_root = os.path.join('checkpoints', opt.name)
if not os.path.isdir(save_root):
    os.makedirs(save_root)

# get the data
climate_data = ClimateDataset(opt=opt)
climate_data_loader = DataLoader(climate_data,
                                 batch_size=opt.batch_size,
                                 shuffle=not opt.no_shuffle,
                                 num_workers=int(opt.n_threads))

# load the model
edgan_model = Edgan(opt=opt)
optimizer = torch.optim.Adam(edgan_model.parameters(), lr=opt.lr)  # TODO which optimizer / lr / lr decay

for epoch in range(opt.n_epochs):
    train_loss = 0
    edgan_model.train()
    for batch_idx, data in enumerate(climate_data_loader, 0):
        fine_pr = data['fine_pr'].to(device)
        coarse_pr = data['coarse_pr'].to(device)
        cell_area = data['cell_area'].to(device)
        orog = data['orog'].to(device)

        optimizer.zero_grad()
        recon_x, mu, log_var = edgan_model(fine_pr=fine_pr, coarse_pr=coarse_pr, orog=orog)
        bce, kld, cycle_loss, loss = edgan_model.loss_function(recon_x, fine_pr, mu, log_var,
                                                               coarse_pr, cell_area)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # todo add visualizer class
        if batch_idx % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBCE Loss: {:.7f}\tKL Loss: {:.7f}\tcycle Loss {:.7f}'
                  '\tLoss: {:.7f}'.format(
                    epoch, batch_idx * len(fine_pr), len(climate_data_loader.dataset),
                    100. * batch_idx / len(climate_data_loader),
                    bce.item() / len(fine_pr),
                    kld.item() / len(fine_pr),
                    cycle_loss.item() / len(fine_pr),
                    loss.item() / len(fine_pr)))
        # todo make logging cluster ready
        if batch_idx % opt.plot_interval == 0:
            vmin = opt.threshold
            vmax = 2
            plt.imshow(fine_pr[0].view(8, 8).detach().numpy(), vmin=vmin, vmax=vmax)
            plt.show()
            plt.imshow(recon_x[0].view(8, 8).detach().numpy(), vmin=vmin, vmax=vmax)
            plt.show()
            # todo make plotting cluster ready

    if epoch % opt.save_interval == 0:
        save_name = "epoch_{}.pth".format(epoch)
        save_dir = os.path.join(save_root, save_name)
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(edgan_model.module.cpu().state_dict(), save_dir)
            edgan_model.cuda(opt.gpu_ids[0])
        else:
            torch.save(edgan_model.cpu().state_dict(), save_dir)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(climate_data_loader.dataset)))
