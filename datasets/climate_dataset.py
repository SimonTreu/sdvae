from torch.utils.data import Dataset
import os.path
import torch
import random
import time
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
        start_time = time.time()
        sample_path = self.sample_paths[item]
        sample = torch.load(sample_path)

        # random crop
        w = sample.shape[-2]
        h = sample.shape[-1]
        w_offset = random.randint(self.fine_size, max(0, w - 2 * self.fine_size - 1))
        h_offset = random.randint(self.fine_size, max(0, h - 2 * self.fine_size - 1))

        # todo Normalize wind
        # normalize pr
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

        return {'fine_pr': fine_pr, 'file_path': sample_path, 'coarse_pr': coarse_pr,
                'cell_area': cell_area, 'orog': orog, 'uas': uas, 'vas': vas, 'time': end_time,
                'coarse_ul': coarse_ul, 'coarse_u': coarse_u, 'coarse_ur': coarse_ur,
                'coarse_l': coarse_l, 'coarse_r': coarse_r,
                'coarse_bl': coarse_bl, 'coarse_b': coarse_b, 'coarse_br': coarse_br, }


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
