import torch


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
        return torch.matmul(torch.matmul(self.s, val), self.s.t())/self.scale_factor**2
