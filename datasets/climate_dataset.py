from torch.utils.data import Dataset


class ClimateDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
