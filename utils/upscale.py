import torch
import numpy as np


def get_average(input_sample, cell_area):
    if input_sample.shape != cell_area.shape:
        raise ValueError("input_sample.shape = {} and "
                         "cell_area.shape = {} are"
                         "not equal".format(input_sample.shape,
                                            cell_area.shape))

    return torch.sum(torch.mul(input_sample, cell_area), dim=-1).div(torch.sum(cell_area, dim=-1))


def upscale(cell_area, scale_factor, upscaling_vars, orog):
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
