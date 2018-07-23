from datasets.climate_dataset import ClimateDataset


class Arg:
    # Todo do that with an arg_parser
    def __init__(self):
        self.dataroot = "data/wind"
        self.phase = "train"
        self.scale_factor = 8


args = Arg()

climate_data = ClimateDataset(opt=args)
