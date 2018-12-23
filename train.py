from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
from models.gamma_vae import GammaVae
from utils.visualizer import Visualizer

import torch
import os
from options.base_options import BaseOptions
import time
import numpy as np


def save(epoch, save_root, gpu_ids, edgan_model):
    save_name = "epoch_{}.pth".format(epoch)
    save_dir = os.path.join(save_root, save_name)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(edgan_model.state_dict(), save_dir)
    else:
        torch.save(edgan_model.state_dict(), save_dir)


def main():
    opt = BaseOptions().parse()
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 else "cpu")

    # create save dir
    save_root = os.path.join('checkpoints', opt.name)
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    # get the data
    climate_data = ClimateDataset(opt=opt, phase='train')
    climate_data_loader = DataLoader(climate_data,
                                     batch_size=opt.batch_size,
                                     shuffle=True,
                                     num_workers=int(opt.n_threads))
    val_data = ClimateDataset(opt=opt, phase='val')
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size,shuffle=True,num_workers=int(opt.n_threads))

    # load the model
    if opt.model == "mse_vae":
        model = Edgan(opt=opt, device=device).to(device)
    elif opt.model == "gamma_vae":
        model = GammaVae(opt=opt, device=device).to(device)
    else:
        raise ValueError("model {} is not implemented".format(opt.model))

    initial_epoch = 0
    if opt.load_epoch >= 0:
        save_name = "epoch_{}.pth".format(opt.load_epoch)
        save_dir = os.path.join(save_root, save_name)
        model.load_state_dict(torch.load(save_dir))
        initial_epoch = opt.load_epoch + 1

    if opt.phase == 'train':
        # get optimizer
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        viz = Visualizer(opt, n_images=5,
                         training_size=len(climate_data_loader.dataset), n_batches=len(climate_data_loader))

        for epoch_idx in range(opt.n_epochs):
            mse_list = []
            kld_list = []
            cycle_loss_list = []
            loss_list = []

            epoch = initial_epoch + epoch_idx
            img_id = 0

            # timing
            epoch_start_time = time.time()
            iter_data_start_time = time.time()
            iter_data_time = 0
            iter_time = 0

            for batch_idx, data in enumerate(climate_data_loader, 0):
                iter_start_time = time.time()

                optimizer.zero_grad()

                if opt.model == "mse_vae": # todo use the same model
                    recon_pr, mu, log_var = model(fine_pr=data['fine_pr'].to(device),
                                                  coarse_pr=data['coarse_pr'].to(device),
                                                  orog=data['orog'].to(device),
                                                  coarse_uas=data['coarse_uas'].to(device),
                                                  coarse_vas=data['coarse_vas'].to(device))

                    mse, kld, cycle_loss, loss = model.loss_function(recon_pr, data['fine_pr'].to(device),
                                                                     mu, log_var,
                                                                     data['coarse_pr'].to(device))
                elif opt.model == "gamma_vae":
                    p, alpha, beta, mu, log_var = model(fine_pr=data['fine_pr'].to(device),
                                                        coarse_pr=data['coarse_pr'].to(device),
                                                        orog=data['orog'].to(device),
                                                        coarse_uas=data['coarse_uas'].to(device),
                                                        coarse_vas=data['coarse_vas'].to(device))

                    mse, kld, cycle_loss, loss = model.loss_function(p, alpha, beta, data['fine_pr'].to(device),
                                                                     mu, log_var,
                                                                     data['coarse_pr'].to(device))
                else:
                    raise ValueError("model {} is not implemented".format(opt.model))

                if loss.item() < float('inf'):
                    loss.backward()

                    mse_list += [mse.item()]
                    kld_list += [kld.item()]
                    cycle_loss_list += [cycle_loss.item()]
                    loss_list += [loss.item()]

                    optimizer.step()
                else:
                    print("inf loss")

                # timing
                iter_time += time.time()-iter_start_time
                iter_data_time += iter_start_time-iter_data_start_time

                if batch_idx % opt.log_interval == 0 and batch_idx > 0:
                    viz.print(epoch, batch_idx,
                              np.mean(mse_list[-opt.log_interval:]),
                              np.mean(kld_list[-opt.log_interval:]),
                              np.mean(cycle_loss_list[-opt.log_interval:]),
                              np.mean(loss_list[-opt.log_interval:]),
                              iter_time,
                              iter_data_time, sum(data['time']).item())

                    iter_data_time = 0
                    iter_time = 0

                if batch_idx % opt.plot_interval == 0 and batch_idx > 0:
                    img_id += 1
                    image_name = "Epoch{}_Image{}.jpg".format(epoch, img_id)
                    if opt.model == "mse_vae":
                        viz.plot(fine_pr=data['fine_pr'].to(device), recon_pr=recon_pr, image_name=image_name)
                    elif opt.model == "gamma_vae":
                        viz.plot(fine_pr=data['fine_pr'].to(device), recon_pr=p*alpha*beta, image_name=image_name)
                    else:
                        raise ValueError("model {} is not implemented".format(opt.model))

                if batch_idx % opt.save_latest_interval == 0 and batch_idx > 0:
                    save('latest', save_root, opt.gpu_ids, model)
                    print('saved latest epoch after {} iterations'.format(batch_idx))

                if batch_idx % opt.eval_val_loss == 0 and batch_idx > 0:
                    # switch model to evaluation mode
                    model.eval()
                    # calculate val loss
                    val_loss_sum = np.zeros(4)  # val_mse, val_kld, val_cycle_loss, val_loss
                    inf_losses = 0  # nr of sets where loss was inf
                    for batch_idx, data in enumerate(val_data_loader, 0):
                        p, alpha, beta, mu, log_var = model(fine_pr=data['fine_pr'].to(device),
                                                            coarse_pr=data['coarse_pr'].to(device),
                                                            orog=data['orog'].to(device), 
                                                            coarse_uas=data['coarse_uas'].to(device), 
                                                            coarse_vas=data['coarse_vas'].to(device))
                        val_loss = model.loss_function(p, alpha, beta,data['fine_pr'].to(device),
                                                       mu, log_var,
                                                       data['coarse_pr'].to(device))
                        val_loss = [l.item() for l in val_loss]
                        if val_loss[-1] < float('inf'):
                            val_loss_sum += val_loss
                        else:
                            inf_losses += 1
                        if batch_idx >= 200:  # todo find a good break value, make shure that it is differen
                            break

                    n_val = 200 - inf_losses
                    viz.print_eval(epoch=epoch,
                                   val_mse=val_loss_sum[0]/n_val,
                                   val_kld=val_loss_sum[1]/n_val,
                                   val_cycle_loss=val_loss_sum[2]/n_val,
                                   val_loss=val_loss_sum[3]/n_val,
                                   inf_losses=inf_losses,
                                   train_mse=np.mean(mse_list[-opt.eval_val_loss:]),
                                   train_kld=np.mean(kld_list[-opt.eval_val_loss:]),
                                   train_cycle_loss=np.mean(cycle_loss_list[-opt.eval_val_loss:]),
                                   train_loss=np.mean(loss_list[-opt.eval_val_loss:]))

                    model.train()



                iter_data_start_time = time.time()

            if epoch % opt.save_interval == 0:
                save(epoch, save_root, opt.gpu_ids, model)
            else:
                print('val inf loss')
            epoch_time = time.time() - epoch_start_time
            viz.print_epoch(epoch=epoch,
                            epoch_mse=np.mean(mse_list),
                            epoch_kld=np.mean(kld_list),
                            epoch_cycle_loss=np.mean(cycle_loss_list),
                            epoch_loss=np.mean(loss_list),
                            epoch_time=epoch_time)


if __name__ == '__main__':
    main()
