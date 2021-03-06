import argparse
import os.path
import torch
from utils import util
from netCDF4 import Dataset


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--name', type=str, default='vae_08_30')
        self.parser.add_argument('--phase', type=str, default="train", help="train, val, test, etc")
        self.parser.add_argument('--batch_size', type=int, default=124, help='input batch size')
        self.parser.add_argument('--n_threads', type=int, default=4, help='# threads for loading data')
        self.parser.add_argument('--nz', type=int, default=2, help='size of bottleneck')
        self.parser.add_argument('--gpu_ids', type=str, default='-1',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        self.parser.add_argument('--n_epochs', type=int, default=20,
                                 help='number of epochs (one epoch is training the hole data one time)')
        self.parser.add_argument('--log_interval', type=int, default=100,
                                 help='number of iterations until the next logging of cost values')
        self.parser.add_argument('--plot_interval', type=int, default=1000000,
                                 help='number of iterations until the next plotting of training results')
        self.parser.add_argument('--eval_val_loss', type=int, default=100,
                                 help='number of iterations until validation loss is calculated')
        self.parser.add_argument('--save_interval', type=int, default=1, help='every _ epoch the model is saved')
        self.parser.add_argument('--save_latest_interval', type=int, default=100, help='every _ iteration the model is saved')
        self.parser.add_argument('--nf_encoder', type=int, default=16,
                                 help='number of filters in first conv layer of encoder')
        self.parser.add_argument('--nf_decoder', type=int, default=16,
                                 help='number of filters in first conv layer of decoder')
        self.parser.add_argument('--load_epoch', type=int, default=-1,
                                 help="if >= 0 load a pre-trained model at the defined epoch, "
                                      "if -1 and training start a new model"
                                      "if -1 and val/test load latest model")
        self.parser.add_argument('--no_orog', action='store_true', help="if specified, don't use topography")
        self.parser.add_argument('--no_dropout', action='store_true', help="if specified, don't use dropout")
        self.parser.add_argument('--n_test', type=int, default=2, help="n test sets in one cell_sized row of lats")
        self.parser.add_argument('--n_val', type=int, default=2, help="n val sets in one cell_sized row of lats")
        self.parser.add_argument('--seed', type=int, default=0, help="seed value for selection of test sets. "
                                                                     "With a different seed,"
                                                                     "different test sets are created")
        self.parser.add_argument('--model', type=str, default='mse_vae', help="which model to use. Available options are"
                                                                              " 'mse_vae' and 'gamma_vae")
        self.parser.add_argument('--regression', action='store_true', help="if specified always use mu= 0,logvar = 0 as output of encoder")

    def parse(self, args=None):
        if args:
            opt = self.parser.parse_args(args)
        else:
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
                            opt_file.write('--{}\n'.format(str(k)))
                    else:
                        opt_file.write('--%s %s\n' % (str(k), str(v)))
                opt_file.write('-------------- Val Option ----------------\n')
                for k, v in sorted(args.items()):
                    if type(v) == bool:
                        if v:
                            opt_file.write('--{}\n'.format(str(k)))
                    elif k == 'load_epoch':
                        opt_file.write('--{} {}\n'.format(str(k), opt.n_epochs-1))
                    elif k == 'phase':
                        opt_file.write('--{} {}\n'.format(str(k), 'val'))
                    elif k == 'gpu_ids':
                        opt_file.write('--{} {}\n'.format(str(k), -1))
                    else:
                        opt_file.write('--%s %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        # parse gpu_ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                opt.gpu_ids.append(int_id)
        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        opt.dataroot = 'data/wind_psl'
        opt.fine_size= 32
        opt.lr = 1e-3  # learning rate
        opt.scale_factor = 8
        opt.n_samples=3  # number of downscaled samples created for each cell in test
        opt.results_dir='./results/'

        return opt
