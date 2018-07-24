from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
import torch
import matplotlib.pyplot as plt


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
        self.cuda = False
        self.n_epochs = 10
        self.log_interval = 10
        self.plot_interval = 250
        self.lambda_cycle_l1 = 10


args = Arg()
device = torch.device("cuda" if args.cuda else "cpu")

# get the data
climate_data = ClimateDataset(opt=args)
climate_data_loader = DataLoader(climate_data,
                                 batch_size=args.batch_size,
                                 shuffle=not args.no_shuffle,
                                 num_workers=int(args.n_threads))

# load the model
edgan_model = Edgan(opt=args)
# todo which optimizer?
optimizer = torch.optim.Adam(edgan_model.parameters(), lr=1e-3)

for epoch in range(args.n_epochs):
    train_loss = 0
    edgan_model.train()
    for batch_idx, data in enumerate(climate_data_loader, 0):
        input_sample = data['input_sample'].to(device)
        average_value = data['average_value'].to(device)
        cell_area = data['cell_area'].to(device)

        optimizer.zero_grad()
        recon_x, mu, log_var = edgan_model(input_sample)
        bce, kld, cycle_loss, loss = edgan_model.loss_function(recon_x, input_sample, mu, log_var,
                                                               average_value, cell_area)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBCE Loss: {:.5f}\tKL Loss: {:.5f}\tcycle Loss {:.5f}'
                  '\tLoss: {:.5f}'.format(
                    epoch, batch_idx * len(input_sample), len(climate_data_loader.dataset),
                    100. * batch_idx / len(climate_data_loader),
                    bce.item() / len(input_sample),
                    kld.item() / len(input_sample),
                    cycle_loss.item(),
                    loss.item() / len(input_sample)))
        if batch_idx % args.plot_interval == 0:
            vmin = 0
            vmax = 1e-3
            plt.imshow(input_sample[0].view(8, 8).detach().numpy(), vmin=vmin, vmax=vmax)
            plt.show()
            plt.imshow(recon_x[0].view(8, 8).detach().numpy(), vmin=vmin, vmax=vmax)
            plt.show()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(climate_data_loader.dataset)))
