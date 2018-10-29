import torch
import numpy as np


def upscale(cell_area, scale_factor, upscaling_vars, orog):
    # todo still necessary? only used in val
    results = []
    n = upscaling_vars[0].shape[-1]
    s = torch.Tensor(np.kron(np.eye(n // scale_factor), np.ones((scale_factor, scale_factor))))
    areas_low_res = torch.matmul(s, torch.matmul(cell_area, s))

    for var in upscaling_vars:
        if not var.shape[-1]%scale_factor == 0:
            raise ValueError("Both dimensions (){} of input must"
                            " be divisible by the scale factor {}".format(var.shape, scale_factor))
        if not var.shape[-1] == var.shape[-2]:
            raise ValueError("input must be a square matrix")

        var = var * cell_area
        var = torch.matmul(s, torch.matmul(var, s))/areas_low_res
        results.append(var)

    results.append(orog)
    return torch.stack(results)


class Upscale:
    def __init__(self, size, scale_factor, device=torch.device("cpu")):
        if size % scale_factor != 0:
            raise ValueError("size must be divisible "
                             "by scale_factor, size={}, scale_factor={}".format(size, scale_factor))
        self.size = size
        self.s = torch.zeros((size // scale_factor, size), dtype=torch.float, device=device)
        for i in range(size // scale_factor):
            for j in range(i*scale_factor, (i+1)*scale_factor):
                self.s[i,j] = 1
        self.scale_factor = scale_factor

    def upscale(self, val):
        for n in val.shape[-2:]:
            if not n == self.size:
                raise ValueError("val must be a square matrix "
                                 "shape=({0}, {0}). But shape is {1}".format(self.size, val.shape))
        return torch.matmul(torch.matmul(self.s, val), torch.t(self.s))/self.scale_factor**2
