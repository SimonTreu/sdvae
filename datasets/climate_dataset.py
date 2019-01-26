from torch.utils.data import Dataset
import os.path
import torch
import random
import time
from utils.upscale import Upscale
from netCDF4 import Dataset as Nc4Dataset


class ClimateDataset(Dataset):
    def __init__(self, opt, phase):
        self.root = opt.dataroot
        self.scale_factor = opt.scale_factor
        self.fine_size = opt.fine_size
        self.n_test = opt.n_test
        self.n_val = opt.n_val
        # cell size is the size of one cell that is extracted from the netcdf file. Afterwards this
        # is cropped to fine_size
        self.cell_size = opt.fine_size + self.scale_factor
        with Nc4Dataset(os.path.join(self.root, "dataset.nc4"), "r", format="NETCDF4") as file:
            check_dimensions(file, self.cell_size, self.scale_factor)
            times = file['time'].size
            cols = file['lon'].size//self.cell_size
            self.rows = file['lat'].size//self.cell_size
            if phase == 'train':
                self.length = self.rows*(cols-self.n_test-self.n_val)*times
            elif phase == 'val':
                self.length = self.rows*self.n_val*times
            elif phase == 'test':
                self.length = self.rows * self.n_test * times
        self.upscaler = Upscale(size=self.fine_size+2*self.scale_factor, scale_factor=self.scale_factor)

        # remove 4 40 boxes to create a training set for each block of 40 rows
        self.lat_lon_list = create_lat_lon_indices(self.rows, cols, self.n_test, self.n_val,
                                                   seed=opt.seed, phase=phase)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start_time = time.time() # todo remove timing

        # ++ calculate lat lon and time from index ++ #
        s_lat = len(self.lat_lon_list)
        s_lon = len(self.lat_lon_list[0])

        t = index // (s_lat * s_lon)
        lat = index % (s_lat * s_lon) // s_lon
        lon = index % s_lon
        # --------------------------------------------------------------------------------------------------------------
        # calculate a random offset from the upper left corner to crop the box
        w_offset = random.randint(0, self.scale_factor-1)
        h_offset = random.randint(0, self.scale_factor-1)
        # --------------------------------------------------------------------------------------------------------------
        # calculate the lat and lon indices of the upper left corner in the netcdf file
        # add scale factor because first 8 pixels are only for boundary conditions
        # --> for lat=0 the index in the netcdf file is 8.
        anchor_lat = lat * self.cell_size + w_offset + self.scale_factor
        anchor_lon = self.lat_lon_list[lat][lon] * self.cell_size + h_offset

        # select indices for a 48 x 48 box around the 32 x 32 box to be downscaled (with boundary values)
        boundary_lats = [i for i in range(anchor_lat-self.scale_factor, anchor_lat+self.fine_size+self.scale_factor)]
        # longitudes might cross the prime meridian
        boundary_lons = [i % 720
                         for i in range(anchor_lon-self.scale_factor, anchor_lon+self.fine_size+self.scale_factor)]
        # --------------------------------------------------------------------------------------------------------------
        # ++ read data ++ #
        with Nc4Dataset(os.path.join(self.root, "dataset.nc4"), "r", format="NETCDF4") as file:
            pr = torch.tensor(file['pr'][t, boundary_lats, boundary_lons], dtype=torch.float)
            orog = torch.tensor(file['orog'][boundary_lats, boundary_lons],  dtype=torch.float)
            uas = torch.tensor(file['uas'][t, boundary_lats, boundary_lons], dtype=torch.float)
            vas = torch.tensor(file['vas'][t, boundary_lats, boundary_lons], dtype=torch.float)
            psl = torch.tensor(file['psl'][t, boundary_lats, boundary_lons], dtype=torch.float)
        # --------------------------------------------------------------------------------------------------------------
        coarse_pr = self.upscaler.upscale(pr)
        coarse_uas = self.upscaler.upscale(uas)
        coarse_vas = self.upscaler.upscale(vas)
        coarse_psl = self.upscaler.upscale(psl)

        # bring all into shape [C,W,H] (Channels, With, Height)
        pr.unsqueeze_(0)
        orog.unsqueeze_(0)
        coarse_pr.unsqueeze_(0)
        coarse_uas.unsqueeze_(0)
        coarse_vas.unsqueeze_(0)
        coarse_psl.unsqueeze_(0)

        end_time = time.time() - start_time

        return {'fine_pr': pr, 'coarse_pr': coarse_pr,
                'orog': orog, 'coarse_uas': coarse_uas,
                'coarse_vas': coarse_vas, 'coarse_psl': coarse_psl,
                'time': end_time}


def check_dimensions(file, cell_size, scale_factor):
    if file['lat'].size % cell_size != 2*scale_factor:
        raise ValueError("Input file dim(lat) must be a multiple of the cell_size + 2*scale_factor "
                         "for the boundary conditions")


def create_lat_lon_indices(rows, cols, n_test, n_val, seed, phase):
    rand = random.Random()
    rand.seed(seed)
    train_indices = [[i for i in range(cols)] for _ in range(rows)]
    val_indices = [rand.sample(train_indices[j], n_val) for j in range(rows)]
    # remove val_indices from train_indices
    train_indices = [[i for i in train_indices[j] if not i in val_indices[j]] for j in range(rows)]
    test_indices = [rand.sample(train_indices[j], n_test) for j in range(rows)]
    # remove test_indices from train_indices
    train_indices = [[i for i in train_indices[j] if not i in test_indices[j]] for j in range(rows)]
    if phase == 'train':
        return train_indices
    elif phase == 'test':
        return test_indices
    elif phase == 'val':
        return val_indices
    else:
        raise ValueError("{} is not a valid argument for phase".format(phase))
