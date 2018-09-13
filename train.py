from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
from utils.visualizer import Visualizer

import torch
import os
from options.base_options import BaseOptions
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
            recon_pr, mu, log_var = edgan_model(fine_pr=fine_pr, coarse_pr=coarse_pr,
                                                coarse_ul=data['coarse_ul'].to(device),
                                                coarse_u=data['coarse_u'].to(device),
                                                coarse_ur=data['coarse_ur'].to(device),
                                                coarse_l=data['coarse_l'].to(device),
                                                coarse_r=data['coarse_r'].to(device),
                                                coarse_bl=data['coarse_bl'].to(device),
                                                coarse_b=data['coarse_b'].to(device),
                                                coarse_br=data['coarse_br'].to(device),
                                                orog=orog)
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


    pass
# TODO normalize all input data with the area weights

# TODO input log(pr + 1) normalized
# TODO what normalization for orog
