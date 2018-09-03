import argparse
import os.path
import torch
from utils import util
from netCDF4 import Dataset


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--name', type=str, default='vae_08_30')
        self.parser.add_argument('--dataroot', type=str, default="data/wind",
                                 help="path to images (should have subfolders trainA, trainB, valA, valB, etc)")
        self.parser.add_argument('--phase', type=str, default="train", help="train, val, test, etc")
        self.parser.add_argument('--fine_size', type=int, default=8,
                                 help="number of high-resolution grid cells within one coarse resolution cell")
        self.parser.add_argument('--batch_size', type=int, default=124, help='input batch size')
        self.parser.add_argument('--no_shuffle', action='store_true',
                                 help='if specified, do not shuffle the input data for batches')
        self.parser.add_argument('--n_threads', type=int, default=4, help='# threads for loading data')
        self.parser.add_argument('--nz', type=int, default=2, help='size of bottleneck')

        self.parser.add_argument('--no', type=int, default=8,
                                 help='size of encoded orography set 0 if non should be used')
        self.parser.add_argument('--gpu_ids', type=str, default='-1',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        self.parser.add_argument('--n_epochs', type=int, default=20,
                                 help='number of epochs (one epoch is training the hole data one time)')
        self.parser.add_argument('--log_interval', type=int, default=100,
                                 help='number of iterations until the next logging of cost values')
        self.parser.add_argument('--plot_interval', type=int, default=1000000,
                                 help='number of iterations until the next plotting of training results')
        self.parser.add_argument('--lambda_cycle_l1', type=int, default=1000,
                                 help='factor to be multiplied with the reconstruction loss '
                                      '(|coarse(reconstructed fine resolution) - coarse input|)')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for optimizer')
        self.parser.add_argument('--save_interval', type=int, default=1, help='every _ epoch the model is saved')
        self.parser.add_argument('--d_hidden', type=int, default=4,
                                 help='number of filters in first conv layer ov encoder')
        self.parser.add_argument('--load_epoch', type=int, default=-1,
                                 help="if >= 0 load a pretrained model at the defined epoch")

    def parse(self):
        opt = self.parser.parse_args()

        # plot selected options
        args = vars(opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            if type(v) == bool and v:
                if v:
                    print(str(k))
            else:
                print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if opt.phase == 'train':
            expr_dir = os.path.join('checkpoints', opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            write_mode = 'w' if opt.load_epoch < 0 else 'a'
            with open(file_name, write_mode) as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    if type(v) == bool:
                        if v:
                            opt_file.write(str(k))
                    else:
                        opt_file.write('--%s %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        # parse gpu_ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                opt.gpu_ids_.append(int_id)
        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # load normalization values
        with Dataset(os.path.join(opt.dataroot, "stats", "mean.nc4"), "r", format="NETCDF4") as rootgrp:
            mean = float(rootgrp.variables['pr'][:])

        with Dataset(os.path.join(opt.dataroot, "stats", "std.nc4"), "r", format="NETCDF4") as rootgrp:
            std = float(rootgrp.variables['pr'][:])

        opt.mean_std = {'mean': mean, 'std': std}
        opt.threshold = (0 - opt.mean_std['mean']) / opt.mean_std['std']

        return opt
