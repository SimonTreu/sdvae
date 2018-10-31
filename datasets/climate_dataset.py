from torch.utils.data import Dataset
import os.path
import torch
import random
import time
from utils.upscale import Upscale
from netCDF4 import Dataset as Nc4Dataset


class ClimateDataset(Dataset):
    def __init__(self, opt):
        self.root = opt.dataroot
        self.scale_factor = opt.scale_factor
        self.fine_size = opt.fine_size
        n_test = opt.n_test
        n_val = opt.n_val
        # cell size is the size of one cell that is extracted from the netcdf file. Afterwards this
        # is croped to fine_size
        self.cell_size = opt.fine_size + self.scale_factor
        with Nc4Dataset(os.path.join(self.root, "dataset.nc4"), "r", format="NETCDF4") as file:
            check_dimensions(file, self.cell_size, self.scale_factor)
            times = file['time'].size
            cols = file['lon'].size//self.cell_size
            rows = file['lat'].size//self.cell_size
            self.length = rows*(cols-n_test-n_val)*times
        self.upscaler = Upscale(size=self.fine_size+2*self.scale_factor, scale_factor=self.scale_factor)

        # remove 4 40 boxes to create a training set for each block of 40 rows
        self.lat_lon_train = [[i for i in range(cols)] for _ in range(rows)]
        self.test_val_indices = create_test_and_val_indices(rows, cols, n_test, n_val, seed=opt.seed)

        #remove_test_sets(self.lat_lon_train)
        #remove_val_sets(self.lat_lon_train)
        # lat = 30 deg north
        self.lat_lon_train = [[self.lat_lon_train[i][j] for j in range(cols) if j not in self.test_val_indices[i]]
                              for i in range(rows)]
        # todo also use that in val

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
        w_offset = random.randint(0, self.scale_factor-1)  # calculate a random offset from the upper left corner to crop the box
        h_offset = random.randint(0, self.scale_factor-1)

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


def check_dimensions(file, cell_size, scale_factor):
    if file['lat'].size % cell_size != 2*scale_factor:
        raise ValueError("Input file dim(lat) must be a multiple of the cell_size + 2*scale_factor "
                         "for the boundary conditions")


def create_test_and_val_indices(rows, cols, n_test, n_val, seed):
    rand = random.Random()
    rand.seed(seed)
    test_val_indices = [rand.sample([i for i in range(cols)], n_val+n_test) for _ in range(rows)]
    return test_val_indices
