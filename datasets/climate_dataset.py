from torch.utils.data import Dataset
import os.path

TORCH_EXTENSION = [
    '.pt'
]


# Dataset implementation
class ClimateDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_samples = os.path.join(opt.dataroot, opt.phase)
        self.sample_paths = sorted(get_sample_files(self.dir_samples))
        self.scale_factor = opt.scale_factor
        # self.norm_parameters = opt.mean_std Todo implement getting the norm parameters while reading the arguments

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, item):
        pass


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
