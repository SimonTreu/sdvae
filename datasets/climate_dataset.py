from torch.utils.data import Dataset
import os.path
import torch
from random import randint
import time
from utils.upscale import Upscale
from netCDF4 import Dataset as Nc4Dataset


TORCH_EXTENSION = [
    '.pt'
]


# Dataset implementation
class ClimateDataset(Dataset):
    def __init__(self, opt):
        self.root = opt.dataroot
        self.scale_factor = opt.scale_factor
        self.fine_size = opt.fine_size  # todo fix in option
        self.cell_size = opt.fine_size + self.scale_factor
        with Nc4Dataset(os.path.join(self.root, "dataset.nc4"), "r", format="NETCDF4") as file:
            times = file['time'].size
        # Number of 40x40 lat lon cells without test and validation cells * number of times
        self.length = (18-4)*3*times
        self.upscaler = Upscale(size=self.fine_size+2*self.scale_factor, scale_factor=self.scale_factor)

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

    def __getitem__(self, index):
        start_time = time.time()

        # calculate indices
        s_lat = len(self.lat_lon_train)
        s_lon = len(self.lat_lon_train[0])

        t = index //(s_lat * s_lon)
        lat = index % (s_lat * s_lon) // s_lon
        lon = index % s_lon

        # calculate a random offset from the upper left corner to crop the box
        w_offset = randint(0, self.scale_factor-1)  # calculate a random offset from the upper left corner to crop the box
        h_offset = randint(0, self.scale_factor-1)

        # add scale factor because first 8 pixels are only for boundary conditions --> for lat=0 the index in the netcdf file is 8.
        anchor_lat = lat * self.cell_size + w_offset + self.scale_factor
        anchor_lon = self.lat_lon_train[lat][lon] * self.cell_size + h_offset

        boundary_lats = [i for i in range(anchor_lat - self.scale_factor, anchor_lat + self.fine_size + self.scale_factor)]
        boundary_lons = [i % 720 for i in
                   range(anchor_lon - self.scale_factor, anchor_lon + self.fine_size + self.scale_factor)]

        lats = [i for i in range(anchor_lat, anchor_lat + self.fine_size)]
        lons = [i for i in range(anchor_lon, anchor_lon + self.fine_size)]

        with Nc4Dataset(os.path.join(self.root, "dataset.nc4"), "r", format="NETCDF4") as file:
            pr = torch.tensor(file['pr'][t, boundary_lats, boundary_lons], dtype=torch.float)
            orog = torch.tensor(file['orog'][lats, lons],  dtype=torch.float)
            cell_area = torch.ones_like(orog)
            # todo cell_area should no more be necessary
            uas = torch.tensor(file['uas'][t, boundary_lats, boundary_lons], dtype=torch.float)
            vas = torch.tensor(file['vas'][t, boundary_lats, boundary_lons], dtype=torch.float)

        coarse_pr = self.upscaler.upscale(pr)
        coarse_uas = self.upscaler.upscale(uas)
        coarse_vas = self.upscaler.upscale(vas)

        fine_pr = pr[self.scale_factor:-self.scale_factor, self.scale_factor:-self.scale_factor]

        end_time = time.time() - start_time

        # bring all into shape [C,W,H] (Channels, With, Height)
        fine_pr.unsqueeze_(0)
        orog.unsqueeze_(0)
        coarse_pr.unsqueeze_(0)
        coarse_uas.unsqueeze_(0)
        coarse_vas.unsqueeze_(0)

        #returns: fine_pr 1x32x32, coarse_pr=1x6x6, uas 1x6x6, vas 1x6x6, orog 32x32

        return {'fine_pr': fine_pr, 'coarse_pr': coarse_pr,
                'orog': orog, 'coarse_uas': coarse_uas, 'coarse_vas': coarse_vas,
                'time': end_time}

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
