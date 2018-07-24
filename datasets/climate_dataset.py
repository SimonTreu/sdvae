from torch.utils.data import Dataset
import os.path
import torch
import random
from utils.upscale import get_average

TORCH_EXTENSION = [
    '.pt'
]


# Dataset implementation
class ClimateDataset(Dataset):
    def __init__(self, opt):
        self.root = opt.dataroot
        self.dir_samples = os.path.join(opt.dataroot, opt.phase)
        self.sample_paths = sorted(get_sample_files(self.dir_samples))
        self.fine_size = opt.fine_size
        # self.norm_parameters = opt.mean_std Todo implement getting the norm parameters while reading the arguments

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, item):
        sample_path = self.sample_paths[item]
        sample = torch.load(sample_path)

        # random crop
        w = sample.shape[-2]
        h = sample.shape[-1]
        w_offset = random.randint(0, max(0, w - self.fine_size-1))
        h_offset = random.randint(0, max(0, h - self.fine_size-1))

        input_sample = sample[:, h_offset:h_offset+self.fine_size, w_offset:w_offset+self.fine_size]
        # get precipitation
        cell_area = input_sample[-1]
        input_sample = input_sample[0]

        # TODO normalization

        average_value = get_average(input_sample, cell_area=cell_area)

        return {'input_sample': input_sample, 'input_path': sample_path, 'average_value': average_value}


# Helper Methods
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
