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
        data = data['input_sample'].to(device)
        optimizer.zero_grad()
        recon_x, mu, log_var = edgan_model(data)
        bce, kld, loss = edgan_model.loss_function(recon_x, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBCE Loss: {:.3f}\tKL Loss: {:.3f}\tLoss: {:.3f}'.format(
                epoch, batch_idx * len(data), len(climate_data_loader.dataset),
                100. * batch_idx / len(climate_data_loader),
                bce.item() / len(data),
                kld.item() / len(data),
                loss.item() / len(data)))
        if batch_idx % args.plot_interval == 0:
            vmin=0
            vmax=1e-3
            plt.imshow(data[0].view(8, 8).detach().numpy(),vmin=vmin, vmax=vmax)
            plt.show()
            plt.imshow(recon_x[0].view(8, 8).detach().numpy(),vmin=vmin, vmax=vmax)
            plt.show()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(climate_data_loader.dataset)))