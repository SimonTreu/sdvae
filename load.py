from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
import torch
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os.path
from scipy.stats import norm
import numpy as np
from matplotlib.widgets import Slider


# TOdo add that back to train.py
class Arg:
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
        self.n_epochs = 10
        self.log_interval = 10
        self.plot_interval = 250
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
        self.load_epoch = 0


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

save_name = "epoch_{}.pth".format(args.load_epoch)
save_dir = os.path.join(save_root, save_name)
edgan_model.load_state_dict(torch.load(save_dir))


# plotting
def get_picture(z_sample, generator):
    # z_sample=norm.ppf(z_sample)
    x_decoded = generator(torch.Tensor(z_sample))
    return x_decoded.view(8,8).detach().numpy()


fig2, ax = plt.subplots()
offset = 0.05*(args.nz + 1)
plt.subplots_adjust(bottom=0.15+offset)

# initial values
z = [0.5 for i in range(args.nz + 1)]


im = get_picture(np.array([z]), generator=edgan_model.decode)
img_in_plot = plt.imshow(im, origin='lower', cmap='viridis')
# position of the slider
z_axis = [plt.axes([0.25, 0.05+i_offset, 0.65, 0.03]) for i_offset in np.arange(offset, 0.0, -0.05)]
z_sliders = [Slider(z_axis[i], 'Z {}'.format(i), -15, 15, valinit=z[i]) for i in range(args.nz + 1)]

def update(val):
    for i in range(args.nz + 1):
        z[i] = z_sliders[i].val
    im = get_picture(np.array([z]), generator=edgan_model.decode)
    img_in_plot.set_data(im)
    fig2.canvas.draw_idle()
    plt.draw()

for z_slider in z_sliders:
    z_slider.on_changed(update)

fig2.show()
plt.show()