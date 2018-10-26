from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
from utils.visualizer import ValidationViz
from netCDF4 import Dataset
from utils.util import make_netcdf_dataset
from utils.upscale import upscale
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy as np

import torch
import os
from options.base_options import BaseOptions
from utils.validate_distribution import ValObj, save_val_data
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
edgan_model = Edgan(opt=opt).to(device)
initial_epoch = 0
if opt.load_epoch >= 0:
    save_name = "epoch_{}.pth".format(opt.load_epoch)
    save_dir = os.path.join(save_root, save_name)
    edgan_model.load_state_dict(torch.load(save_dir))
    initial_epoch = opt.load_epoch + 1

# Iterate the netcdf files and plot reconstructed images.
dir_samples = os.path.join(opt.dataroot, 'nc4_data', opt.phase)
nc4_sample_paths = sorted(make_netcdf_dataset(dir_samples))
for input_dataset_path in nc4_sample_paths:
    with Dataset(input_dataset_path, "r", format="NETCDF4") as input_dataset:
        prs = torch.Tensor(input_dataset['pr'][:])
        orog = torch.Tensor(input_dataset['orog'][:])
        cell_area = torch.Tensor(input_dataset['cell_area'][:])

        # normalize pr and orog
        prs.sub_(opt.mean_std['mean_pr']).div_(opt.mean_std['std_pr'])
        orog.sub_(opt.mean_std['mean_orog']).div_(opt.mean_std['std_orog'])
        # get a random element:
        r = np.random.randint(0, prs.shape[0])
        # get coarse pr
        fine_pr = prs[r]
        coarse_pr = upscale(cell_area=cell_area,
                            scale_factor=opt.fine_size,
                            upscaling_vars=[fine_pr],
                            orog=orog)[0].to(device)  # only pr
        # get recon pr
        def get_recon():
            recon_pr = torch.zeros((24,24))
            # ul
            recon_pr[:8, :8] = edgan_model.get_picture(coarse_ul=coarse_pr[0, 0].view(1,1,1,1),
                                                       coarse_u=coarse_pr[8, 0].view(1,1,1,1),
                                                       coarse_ur=coarse_pr[16, 0].view(1,1,1,1),
                                                       coarse_l=coarse_pr[0, 8].view(1,1,1,1),
                                                       coarse_precipitation=coarse_pr[8, 8].view(1,1,1,1),
                                                       coarse_r=coarse_pr[16, 8].view(1,1,1,1),
                                                       coarse_bl=coarse_pr[0, 16].view(1,1,1,1),
                                                       coarse_b=coarse_pr[8, 16].view(1,1,1,1),
                                                       coarse_br=coarse_pr[16, 16].view(1,1,1,1),
                                                       orog=orog[8:16, 8:16].view(1,1,8,8))
            # u
            recon_pr[8:16, :8] = edgan_model.get_picture(coarse_ul=coarse_pr[8, 0].view(1,1,1,1),
                                                         coarse_u=coarse_pr[16, 0].view(1,1,1,1),
                                                         coarse_ur=coarse_pr[24, 0].view(1,1,1,1),
                                                         coarse_l=coarse_pr[8, 8].view(1,1,1,1),
                                                         coarse_precipitation=coarse_pr[16, 8].view(1,1,1,1),
                                                         coarse_r=coarse_pr[24, 8].view(1,1,1,1),
                                                         coarse_bl=coarse_pr[8, 16].view(1,1,1,1),
                                                         coarse_b=coarse_pr[16, 16].view(1,1,1,1),
                                                         coarse_br=coarse_pr[24, 16].view(1,1,1,1),
                                                         orog=orog[16:24, 8:16].view(1,1,8,8))

            # ur
            recon_pr[16:24, :8] = edgan_model.get_picture(coarse_ul=coarse_pr[16, 0].view(1,1,1,1),
                                                          coarse_u=coarse_pr[24, 0].view(1,1,1,1),
                                                          coarse_ur=coarse_pr[32, 0].view(1,1,1,1),
                                                          coarse_l=coarse_pr[16, 8].view(1,1,1,1),
                                                          coarse_precipitation=coarse_pr[24, 8].view(1,1,1,1),
                                                          coarse_r=coarse_pr[32, 8].view(1,1,1,1),
                                                          coarse_bl=coarse_pr[16, 16].view(1,1,1,1),
                                                          coarse_b=coarse_pr[24, 16].view(1,1,1,1),
                                                          coarse_br=coarse_pr[32, 16].view(1,1,1,1),
                                                          orog=orog[24:32, 8:16].view(1,1,8,8))

            # l
            recon_pr[:8, 8:16] = edgan_model.get_picture(coarse_ul=coarse_pr[0, 8].view(1,1,1,1),
                                                         coarse_u=coarse_pr[8, 8].view(1,1,1,1),
                                                         coarse_ur=coarse_pr[16, 8].view(1,1,1,1),
                                                         coarse_l=coarse_pr[0, 16].view(1,1,1,1),
                                                         coarse_precipitation=coarse_pr[8, 16].view(1,1,1,1),
                                                         coarse_r=coarse_pr[16, 16].view(1,1,1,1),
                                                         coarse_bl=coarse_pr[0, 24].view(1,1,1,1),
                                                         coarse_b=coarse_pr[8, 24].view(1,1,1,1),
                                                         coarse_br=coarse_pr[16, 24].view(1,1,1,1),
                                                         orog=orog[8:16, 16:24].view(1,1,8,8))

            # c
            recon_pr[8:16, 8:16] = edgan_model.get_picture(coarse_ul=coarse_pr[8, 8].view(1,1,1,1),
                                                           coarse_u=coarse_pr[16, 8].view(1,1,1,1),
                                                           coarse_ur=coarse_pr[24, 8].view(1,1,1,1),
                                                           coarse_l=coarse_pr[8, 16].view(1,1,1,1),
                                                           coarse_precipitation=coarse_pr[16, 16].view(1,1,1,1),
                                                           coarse_r=coarse_pr[24, 16].view(1,1,1,1),
                                                           coarse_bl=coarse_pr[8, 24].view(1,1,1,1),
                                                           coarse_b=coarse_pr[16, 24].view(1,1,1,1),
                                                           coarse_br=coarse_pr[24, 24].view(1,1,1,1),
                                                           orog=orog[16:24, 16:24].view(1,1,8,8))

            # r
            recon_pr[16:24, 8:16] = edgan_model.get_picture(coarse_ul=coarse_pr[16, 8].view(1,1,1,1),
                                                            coarse_u=coarse_pr[24, 8].view(1,1,1,1),
                                                            coarse_ur=coarse_pr[32, 8].view(1,1,1,1),
                                                            coarse_l=coarse_pr[16, 16].view(1,1,1,1),
                                                            coarse_precipitation=coarse_pr[24, 16].view(1,1,1,1),
                                                            coarse_r=coarse_pr[32, 16].view(1,1,1,1),
                                                            coarse_bl=coarse_pr[16, 24].view(1,1,1,1),
                                                            coarse_b=coarse_pr[24, 24].view(1,1,1,1),
                                                            coarse_br=coarse_pr[32, 24].view(1,1,1,1),
                                                            orog=orog[24:32, 16:24].view(1,1,8,8))

            # bl
            recon_pr[:8, 16:24] = edgan_model.get_picture(coarse_ul=coarse_pr[0, 16].view(1,1,1,1),
                                                          coarse_u=coarse_pr[8, 16].view(1,1,1,1),
                                                          coarse_ur=coarse_pr[16, 16].view(1,1,1,1),
                                                          coarse_l=coarse_pr[0, 24].view(1,1,1,1),
                                                          coarse_precipitation=coarse_pr[8, 24].view(1,1,1,1),
                                                          coarse_r=coarse_pr[16, 24].view(1,1,1,1),
                                                          coarse_bl=coarse_pr[0, 32].view(1,1,1,1),
                                                          coarse_b=coarse_pr[8, 32].view(1,1,1,1),
                                                          coarse_br=coarse_pr[16, 32].view(1,1,1,1),
                                                          orog=orog[8:16, 24:32].view(1,1,8,8))

            # b
            recon_pr[8:16, 16:24] = edgan_model.get_picture(coarse_ul=coarse_pr[8, 16].view(1,1,1,1),
                                                            coarse_u=coarse_pr[16, 16].view(1,1,1,1),
                                                            coarse_ur=coarse_pr[24, 16].view(1,1,1,1),
                                                            coarse_l=coarse_pr[8, 24].view(1,1,1,1),
                                                            coarse_precipitation=coarse_pr[16, 24].view(1,1,1,1),
                                                            coarse_r=coarse_pr[24, 24].view(1,1,1,1),
                                                            coarse_bl=coarse_pr[8, 32].view(1,1,1,1),
                                                            coarse_b=coarse_pr[16, 32].view(1,1,1,1),
                                                            coarse_br=coarse_pr[24, 32].view(1,1,1,1),
                                                            orog=orog[16:24, 24:32].view(1,1,8,8))

            # br
            recon_pr[16:24, 16:24] = edgan_model.get_picture(coarse_ul=coarse_pr[16, 16].view(1,1,1,1),
                                                            coarse_u=coarse_pr[24, 16].view(1,1,1,1),
                                                            coarse_ur=coarse_pr[32, 16].view(1,1,1,1),
                                                            coarse_l=coarse_pr[16, 24].view(1,1,1,1),
                                                            coarse_precipitation=coarse_pr[24, 24].view(1,1,1,1),
                                                            coarse_r=coarse_pr[32, 24].view(1,1,1,1),
                                                            coarse_bl=coarse_pr[16, 32].view(1,1,1,1),
                                                            coarse_b=coarse_pr[24, 32].view(1,1,1,1),
                                                            coarse_br=coarse_pr[32, 32].view(1,1,1,1),
                                                            orog=orog[24:32, 24:32].view(1,1,8,8))
            return recon_pr
        # plot a random pr field
        recon_pr = get_recon()
        fig, ax = plt.subplots(1,3)
        offset = plt.subplots_adjust(bottom=0.2)

        vmin = opt.threshold
        vmax = 15
        ax[0].imshow(fine_pr[8:32,8:32], origin='lower',
                                 cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax)
        ax[0].set_title('Fine Pr')
        recon_im = ax[1].imshow(recon_pr, origin='lower',
                                 cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax)
        ax[1].set_title('Recon Pr')

        ax[2].imshow(coarse_pr, origin='lower',
                     cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax)
        ax[2].set_title('Coarse Pr')

        def new_recon(val):
            recon_pr = get_recon()
            recon_im.set_data(recon_pr)
            fig.canvas.draw_idle()
            plt.draw()
        ax_button = plt.axes([0.81, 0.0, 0.1, 0.075])
        b_orog = Button(ax_button, 'New Recon')
        b_orog.on_clicked(new_recon)

        plt.show()

        pass

# Plot Figure to evaluate latent space
viz = ValidationViz(opt)
viz.plot_latent_walker(edgan_model, climate_data)

base_path = os.path.join(save_root, 'epoch_{}_all'.format(opt.load_epoch))
if not os.path.isfile(os.path.abspath(base_path + '_coarse_pr.pt')):
    all_fine_pr = None
    all_recon_pr = None
    all_coarse = None
    for batch_idx, data in enumerate(climate_data_loader, 0):
        fine_pr = data['fine_pr'].to(device)
        coarse_pr = data['coarse_pr'].to(device)
        cell_area = data['cell_area'].to(device)
        orog = data['orog'].to(device)
        recon_pr = edgan_model.get_picture(coarse_precipitation=coarse_pr,
                                           coarse_ul=data['coarse_ul'].to(device),
                                           coarse_u=data['coarse_u'].to(device),
                                           coarse_ur=data['coarse_ur'].to(device),
                                           coarse_l=data['coarse_l'].to(device),
                                           coarse_r=data['coarse_r'].to(device),
                                           coarse_bl=data['coarse_bl'].to(device),
                                           coarse_b=data['coarse_b'].to(device),
                                           coarse_br=data['coarse_br'].to(device),
                                           orog=orog)
        if not(all_fine_pr is None):
            all_fine_pr = torch.cat((all_fine_pr, fine_pr), 0)
            all_recon_pr = torch.cat((all_recon_pr, recon_pr), 0)
            all_coarse = torch.cat((all_coarse, coarse_pr), 0)
        else:
            all_fine_pr = fine_pr
            all_recon_pr = recon_pr
            all_coarse = coarse_pr
    save_val_data(all_fine_pr, all_coarse, all_recon_pr, base_path)

val_obj = ValObj(base_path, min=20, max=50)
val_obj.evaluate_distribution()
pass


