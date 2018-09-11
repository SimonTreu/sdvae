from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
from utils.visualizer import ValidationViz

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
        recon_pr = edgan_model.get_picture(coarse_precipitation=coarse_pr, orog=orog)
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


