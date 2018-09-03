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
        self.norm_parameters = opt.mean_std

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
        orog = input_sample[-2]
        # todo add wind
        fine_pr = input_sample[0]

        # normalize parameters
        fine_pr.sub_(self.norm_parameters['mean']).div_(self.norm_parameters['std'])

        coarse_pr = get_average(fine_pr.contiguous().view(1, -1), cell_area=cell_area.contiguous().view(1, -1))


        return {'fine_pr': fine_pr, 'file_path': sample_path, 'coarse_pr': coarse_pr,
                'cell_area': cell_area, 'orog':orog}


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
