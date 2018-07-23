from datasets.climate_dataset import ClimateDataset


class Arg:
    def __init__(self):
        pass


args = Arg()

climate_data = ClimateDataset(opt=args)
