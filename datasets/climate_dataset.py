from torch.utils.data import Dataset
import os.path
import torch
import random
import time
from utils.upscale import get_average
from netCDF4 import Dataset as Nc4Dataset

TORCH_EXTENSION = [
    '.pt'
]


# Dataset implementation
class ClimateDataset(Dataset):
    def __init__(self, opt):
        self.root = opt.dataroot
        self.scale_factor = opt.scale_factor  # todo add to options
        self.fine_size = opt.fine_size  # todo fix in option
        self.cell_size = opt.fine_size + self.scale_factor
        with Nc4Dataset(os.path.join(self.root, "dataset.nc4"), "r", format="NETCDF4") as file:
            times = file['time'].size
        # Number of 40x40 lat lon cells without test and validation cells * number of times
        self.length = (18-4)*3*times

        # remove 4 40 boxes to create a training set for each block of 40 rows
        self.lat_lon_train = [[i for i in range(18)] for j in range(3)]
        # lat = 30 deg north
        self.lat_lon_train[0].remove(6)   # test
        self.lat_lon_train[0].remove(10)  # test
        self.lat_lon_train[0].remove(15)  # val
        self.lat_lon_train[0].remove(16)  # val
        # lat = 10 deg north
        self.lat_lon_train[1].remove(0)   # test
        self.lat_lon_train[1].remove(4)   # test
        self.lat_lon_train[1].remove(12)  # val
        self.lat_lon_train[1].remove(14)  # val
        # lat = -10 deg north
        self.lat_lon_train[2].remove(1)   # test
        self.lat_lon_train[2].remove(7)   # test
        self.lat_lon_train[2].remove(10)  # val
        self.lat_lon_train[2].remove(13)  # val

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # todo load from netcdf instead
        start_time = time.time()
        sample_path = self.sample_paths[item]
        sample = torch.load(sample_path)

        # random crop
        w = sample.shape[-2]
        h = sample.shape[-1]
        # todo proper offset
        w_offset = random.randint(self.fine_size, max(0, w - 2 * self.fine_size - 1))
        h_offset = random.randint(self.fine_size, max(0, h - 2 * self.fine_size - 1))

        # todo add wind
        # normalize pr
        # todo no normalization
        sample[0].sub_(self.norm_parameters['mean_pr']).div_(self.norm_parameters['std_pr'])
        # normalize orog
        sample[-2].sub_(self.norm_parameters['mean_orog']).div_(self.norm_parameters['std_orog'])

        input_sample = sample[:, h_offset:h_offset+self.fine_size, w_offset:w_offset+self.fine_size]

        sample_ul = sample[:,
                    h_offset+self.fine_size:h_offset+2*self.fine_size,
                    w_offset-self.fine_size:w_offset]
        sample_u = sample[:,
                   h_offset+self.fine_size:h_offset + 2 * self.fine_size,
                   w_offset:w_offset+self.fine_size]
        sample_ur = sample[:,
                    h_offset+self.fine_size:h_offset + 2 * self.fine_size,
                    w_offset+self.fine_size:w_offset+2*self.fine_size]

        sample_l = sample[:,
                   h_offset:h_offset+self.fine_size,
                   w_offset-self.fine_size:w_offset]
        sample_r = sample[:,
                   h_offset:h_offset+self.fine_size,
                   w_offset+self.fine_size:w_offset+2*self.fine_size]

        sample_bl = sample[:,
                    h_offset-self.fine_size:h_offset,
                    w_offset-self.fine_size:w_offset]
        sample_b = sample[:,
                   h_offset-self.fine_size:h_offset,
                   w_offset:w_offset+self.fine_size]
        sample_br = sample[:,
                    h_offset - self.fine_size:h_offset,
                    w_offset + self.fine_size:w_offset + 2 * self.fine_size]

        cell_area = input_sample[-1]
        orog = input_sample[-2]
        fine_pr = input_sample[0]
        pr_ul = sample_ul[0]
        pr_u = sample_u[0]
        pr_ur = sample_ur[0]

        pr_l = sample_l[0]
        pr_r = sample_r[0]

        pr_bl = sample_bl[0]
        pr_b = sample_b[0]
        pr_br = sample_br[0]

        fine_uas = input_sample[1]
        fine_vas = input_sample[2]

        # todo use Upscale class instead


        coarse_pr = get_average(fine_pr.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)  # Shape [C=1, W=1, H=1]

        coarse_ul = get_average(pr_ul.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)
        coarse_u = get_average(pr_u.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)
        coarse_ur = get_average(pr_ur.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)

        coarse_l = get_average(pr_l.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)
        coarse_r = get_average(pr_r.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)

        coarse_bl = get_average(pr_bl.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)
        coarse_b = get_average(pr_b.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)
        coarse_br = get_average(pr_br.contiguous().view(1, -1),
                                cell_area=cell_area.contiguous().view(1, -1)
                                ).unsqueeze(0).unsqueeze(0)


        uas = get_average(fine_uas.contiguous().view(1, -1),
                          cell_area=cell_area.contiguous().view(1, -1)
                          ).unsqueeze(0).unsqueeze(0)
        vas = get_average(fine_vas.contiguous().view(1, -1),
                          cell_area=cell_area.contiguous().view(1, -1)
                          ).unsqueeze(0).unsqueeze(0)

        end_time = time.time() - start_time

        # bring all into shape [C,W,H] (Channels, With, Height)
        fine_pr.unsqueeze_(0)
        orog.unsqueeze_(0)

        #todo intput: fine_pr 32x32, coarse_pr=6x6, uas 6x6, vas 6x6, orog 32x32

        return {'fine_pr': fine_pr, 'file_path': sample_path, 'coarse_pr': coarse_pr,
                'cell_area': cell_area, 'orog': orog, 'uas': uas, 'vas': vas, 'time': end_time,
                'coarse_ul': coarse_ul, 'coarse_u': coarse_u, 'coarse_ur': coarse_ur,
                'coarse_l': coarse_l, 'coarse_r': coarse_r,
                'coarse_bl': coarse_bl, 'coarse_b': coarse_b, 'coarse_br': coarse_br, }


# Helper Methods
# todo remove helper methods here
def is_torch_file(filename):
    return any(filename.endswith(extension) for extension in TORCH_EXTENSION)


def get_sample_files(sample_dir):
    cells = []
    assert os.path.isdir(sample_dir), '%s is not a valid directory' % sample_dir

    for root, _, fnames in sorted(os.walk(sample_dir)):
        for fname in fnames:
            if is_torch_file(fname):
                path = os.path.join(root, fname)
                cells.append(path)

    return cells
