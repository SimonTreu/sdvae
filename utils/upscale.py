import torch


def get_average(input_sample, cell_area):
    if input_sample.shape != cell_area.shape:
        raise ValueError("input_sample.shape = {} and "
                         "cell_area.shape = {} are"
                         "not equal".format(input_sample.shape,
                                            cell_area.shape))
    if len(input_sample.shape) != 2:
        raise ValueError("input sample should be a 2d tensor")
    return torch.sum(torch.mul(input_sample, cell_area)).div(torch.sum(cell_area))
