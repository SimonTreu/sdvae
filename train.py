from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan


class Arg:
    # Todo do that with an arg_parser
    def __init__(self):
        self.dataroot = "data/wind"
        self.phase = "train"
        self.fine_size = 8
        self.batch_size = 124
        self.no_shuffle = False
        self.n_threads = 4
        # Number of hidden variables
        self.nz = 2


args = Arg()

# get the data
climate_data = ClimateDataset(opt=args)
climate_data_loader = DataLoader(climate_data,
                                 batch_size=args.batch_size,
                                 shuffle=not args.no_shuffle,
                                 num_workers=int(args.n_threads))

# load the model
edgan_model = Edgan(opt=args)