import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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
                          iter_data_time, sum(data['time']).item())
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
    vmax = 7

    # plotting
    def get_picture(coarse_pr, generator, z=None, orog=None):
        if orog is None:
            orog = torch.ones(1, 1,  opt.fine_size, opt.fine_size) * opt.threshold

        if z is None:
            z = torch.randn(coarse_pr.shape[0], opt.nz, 1, 1)

        if opt.no > 0:
            o = generator.encode_orog(orog)
            o2 = generator.encode_orog_2(orog)
            x_decoded = generator.decode(z=z, o=o, o2=o2, coarse_pr=coarse_pr)
        else:
            x_decoded = generator.decode(z=z, coarse_pr=coarse_pr)
        if coarse_pr.shape[0] == 1:
            return x_decoded.view(8, 8)
        else:
            return x_decoded

    fig2, ax = plt.subplots()
    offset = 0.05 * (opt.nz + 1)
    plt.subplots_adjust(bottom=0.15 + offset)

    # initial values
    z = torch.ones(1, opt.nz, 1, 1) * 0.5
    coarse_pr = torch.ones(1, 1, 1, 1) * 0.5

    r = np.random.randint(len(climate_data))
    orog = climate_data[r]['orog'].unsqueeze(0)

    im = get_picture(z=torch.Tensor(norm.ppf(z)), coarse_pr=coarse_pr, orog=orog, generator=edgan_model)
    img_in_plot = plt.imshow(im.detach().numpy(), origin='lower', cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax)
    orog_in_plot = plt.contour(orog.view(8,8))
    # position of the slider
    z_axis = [plt.axes([0.25, 0.05 + i_offset, 0.65, 0.03]) for i_offset in np.arange(offset, 0.0, -0.05)]
    z_sliders = [Slider(z_axis[i], 'Z {}'.format(i), 0, 1, valinit=z[0, i, 0, 0].item()) for i in range(opt.nz)]
    z_sliders.append(Slider(z_axis[opt.nz], 'Coarse Pr', opt.threshold, 10, valinit=coarse_pr))


    def update(val):
        for i in range(opt.nz):
            z[0, i, 0, 0] = z_sliders[i].val
        coarse_pr[0, 0, 0, 0] = z_sliders[-1].val
        im = get_picture(z=torch.Tensor(norm.ppf(z)), coarse_pr=coarse_pr, orog=orog, generator=edgan_model)
        img_in_plot.set_data(im.detach().numpy())
        fig2.canvas.draw_idle()
        fig2.suptitle('{}, {}'.format(coarse_pr.item(), torch.mean(im).item()))
        plt.draw()

    def update_orog(val):
        global orog_in_plot
        if opt.no > 0:
            r = np.random.randint(len(climate_data))
            orog = climate_data[r]['orog'].unsqueeze(0)
        im = get_picture(z=torch.Tensor(norm.ppf(z)), coarse_pr=coarse_pr, orog=orog, generator=edgan_model)
        img_in_plot.set_data(im.detach().numpy())
        for coll in orog_in_plot.collections:
            coll.remove()
        orog_in_plot = ax.contour(orog.view(8, 8))
        fig2.canvas.draw_idle()
        fig2.suptitle('{}, {}'.format(coarse_pr.item(), torch.mean(im).item()))
        plt.draw()


    for z_slider in z_sliders:
        z_slider.on_changed(update)

    ax_button = plt.axes([0.81, 0.0, 0.1, 0.075])
    b_orog = Button(ax_button, 'Orog')
    b_orog.on_clicked(update_orog)

    fig2.show()
    plt.show()

    all_fine_pr = None
    all_recon_pr = None
    all_coarse = None
    for batch_idx, data in enumerate(climate_data_loader, 0):
        fine_pr = data['fine_pr'].to(device)
        coarse_pr = data['coarse_pr'].to(device)
        cell_area = data['cell_area'].to(device)
        orog = data['orog'].to(device)
        if not(all_fine_pr is None):
            all_fine_pr = torch.cat((all_fine_pr, fine_pr), 0)
            all_recon_pr = torch.cat((all_recon_pr, get_picture(coarse_pr=coarse_pr, generator=edgan_model, orog=orog)), 0)
            all_coarse = torch.cat((all_coarse, coarse_pr), 0)
        else:
            all_fine_pr = fine_pr
            all_recon_pr = get_picture(coarse_pr=coarse_pr, generator=edgan_model, orog=orog)
            all_coarse = coarse_pr

    pass
# TODO normalize all input data with the area weights