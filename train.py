import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
from utils.visualizer import Visualizer
import torch
import os
from options.base_options import BaseOptions
import numpy as np
from scipy.stats import norm
import time


opt = BaseOptions().parse()
device = torch.device("cuda" if len(opt.gpu_ids) > 0 else "cpu")

# create save dir
save_root = os.path.join('checkpoints', opt.name)
if not os.path.isdir(save_root):
    os.makedirs(save_root)

# get the data
climate_data = ClimateDataset(opt=opt)
climate_data_loader = DataLoader(climate_data,
                                 batch_size=opt.batch_size,
                                 shuffle=not opt.no_shuffle,
                                 num_workers=int(opt.n_threads))

# load the model
edgan_model = Edgan(opt=opt).to(device)
initial_epoch = 0
if opt.load_epoch >= 0:
    save_name = "epoch_{}.pth".format(opt.load_epoch)
    save_dir = os.path.join(save_root, save_name)
    edgan_model.load_state_dict(torch.load(save_dir))
    initial_epoch = opt.load_epoch + 1

if opt.phase == 'train':
    # get optimizer
    edgan_model.train()
    optimizer = torch.optim.Adam(edgan_model.parameters(), lr=opt.lr)  # TODO which optimizer / lr / lr decay
    viz = Visualizer(opt, n_images=5, training_size=len(climate_data_loader.dataset), n_batches=len(climate_data_loader))

    for epoch_idx in range(opt.n_epochs):
        epoch_start_time = time.time()
        epoch = initial_epoch + epoch_idx
        img_id = 0
        epoch_mse = 0
        epoch_kld = 0
        epoch_cycle_loss = 0
        epoch_loss = 0
        iter_data_start_time = time.time()
        iter_data_time = 0
        iter_time = 0
        for batch_idx, data in enumerate(climate_data_loader, 0):
            iter_start_time = time.time()
            fine_pr = data['fine_pr'].to(device)
            coarse_pr = data['coarse_pr'].to(device)
            cell_area = data['cell_area'].to(device)
            orog = data['orog'].to(device)

            optimizer.zero_grad()
            recon_pr, mu, log_var = edgan_model(fine_pr=fine_pr, coarse_pr=coarse_pr, orog=orog)
            mse, kld, cycle_loss, loss = edgan_model.loss_function(recon_pr, fine_pr, mu, log_var,
                                                                   coarse_pr, cell_area)
            loss.backward()

            epoch_mse += mse.item()
            epoch_kld += kld.item()
            epoch_cycle_loss += cycle_loss.item()
            epoch_loss += loss.item()
            optimizer.step()
            iter_time += time.time()-iter_start_time
            iter_data_time += iter_start_time-iter_data_start_time

            if batch_idx % opt.log_interval == 0:
                viz.print(epoch, batch_idx, mse, kld, cycle_loss, loss, iter_time,
                          iter_data_time)
                iter_data_time = 0
                iter_time = 0
            if batch_idx % opt.plot_interval == 0:
                img_id += 1
                image_name = "Epoch{}_Image{}.jpg".format(epoch, img_id)
                viz.plot(fine_pr=fine_pr, recon_pr=recon_pr, image_name=image_name)
            iter_data_start_time = time.time()

        if epoch % opt.save_interval == 0:
            save_name = "epoch_{}.pth".format(epoch)
            save_dir = os.path.join(save_root, save_name)
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(edgan_model.module.cpu().state_dict(), save_dir)
                edgan_model.cuda(opt.gpu_ids[0])
            else:
                torch.save(edgan_model.cpu().state_dict(), save_dir)
        epoch_time = time.time() - epoch_start_time
        print('-----------------------------------------------------------------------------------')
        print('====> Epoch: {}, average MSE: {:.2f}, average KL loss: {:.2f}, '
              'average cycle loss: {:.2f}, average loss: {:.2f}, calculation time = {:.2f}'.format(
               epoch,
               epoch_mse / len(climate_data_loader.dataset),
               epoch_kld / len(climate_data_loader.dataset),
               epoch_cycle_loss / len(climate_data_loader.dataset),
               epoch_loss / len(climate_data_loader.dataset),
               epoch_time))
        print('------------------------------------------------------------------------------------')

    save_name = "epoch_{}.pth".format(epoch)
    save_dir = os.path.join(save_root, save_name)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(edgan_model.module.cpu().state_dict(), save_dir)
        edgan_model.cuda(opt.gpu_ids[0])
    else:
        torch.save(edgan_model.cpu().state_dict(), save_dir)
else:
    vmin = opt.threshold
    vmax = 6
    # plotting
    def get_picture(z_sample, generator, orog=None):
        # todo fix all the formatting stuff (unsqueeze and .view)
        if orog is None:
            orog = torch.ones(1, 1,  opt.fine_size, opt.fine_size) * opt.threshold
        # z_sample=norm.ppf(z_sample)
        z = torch.Tensor(z_sample)
        if opt.no > 0:
            o = generator.encode_orog(orog)
            x_decoded = generator.decode(torch.cat((z, o.view(-1, opt.no)), 1).unsqueeze(-1).unsqueeze(-1))
        else:
            x_decoded = generator.decode(z.unsqueeze(-1).unsqueeze(-1))
        return x_decoded.view(8, 8).detach().numpy()

    fig2, ax = plt.subplots()
    offset = 0.05 * (opt.nz + 1)
    plt.subplots_adjust(bottom=0.15 + offset)

    # initial values
    z = [0.5 for i in range(opt.nz + 1)]

    im = get_picture(np.array([z]), generator=edgan_model)
    img_in_plot = plt.imshow(im, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    # position of the slider
    z_axis = [plt.axes([0.25, 0.05 + i_offset, 0.65, 0.03]) for i_offset in np.arange(offset, 0.0, -0.05)]
    z_sliders = [Slider(z_axis[i], 'Z {}'.format(i), 0, 1, valinit=z[i]) for i in range(opt.nz)]
    z_sliders.append(Slider(z_axis[opt.nz], 'Coarse Pr', opt.threshold, 10, valinit=z[opt.nz]))


    def update(val):
        for i in range(opt.nz + 1):
            z[i] = z_sliders[i].val
        z[:-1] = norm.ppf(z[:-1])
        im = get_picture(np.array([z]), generator=edgan_model)
        img_in_plot.set_data(im)
        fig2.canvas.draw_idle()
        plt.draw()


    for z_slider in z_sliders:
        z_slider.on_changed(update)

    fig2.show()
    plt.show()