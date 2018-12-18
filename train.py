from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
from models.gamma_vae import GammaVae
from utils.visualizer import Visualizer

import torch
import os
from options.base_options import BaseOptions
import time


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
    climate_data = ClimateDataset(opt=opt)
    climate_data_loader = DataLoader(climate_data,
                                     batch_size=opt.batch_size,
                                     shuffle=not opt.no_shuffle,
                                     num_workers=int(opt.n_threads))

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
        viz = Visualizer(opt, n_images=5, training_size=len(climate_data_loader.dataset), n_batches=len(climate_data_loader))

        lambda_kl = opt.lambda_kl
        lambda_kl_update_interval = len(climate_data_loader)//20  # update after 5% of dataset is processed
        lambda_kl_update_rate = (1-lambda_kl)/(20*opt.n_epochs)

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
            interval_mse = []
            interval_kld = []
            interval_cycle_loss = []
            interval_loss = []
            for batch_idx, data in enumerate(climate_data_loader, 0):
                if batch_idx % lambda_kl_update_interval == 0 and batch_idx > 0:
                    lambda_kl += lambda_kl_update_rate
                    print("lambda_kl = {}".format(lambda_kl))
                iter_start_time = time.time()
                fine_pr = data['fine_pr'].to(device)
                coarse_pr = data['coarse_pr'].to(device)
                coarse_uas = data['coarse_uas'].to(device)
                coarse_vas = data['coarse_vas'].to(device)
                orog = data['orog'].to(device)

                optimizer.zero_grad()

                if opt.model == "mse_vae":
                    recon_pr, mu, log_var = model(fine_pr=fine_pr, coarse_pr=coarse_pr,
                                                  orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas)
                    mse, kld, cycle_loss, loss = model.loss_function(recon_pr, fine_pr, mu, log_var,
                                                                     coarse_pr, lambda_kl)
                elif opt.model == "gamma_vae":
                    p, alpha, beta, mu, log_var = model(fine_pr=fine_pr, coarse_pr=coarse_pr,
                                                  orog=orog, coarse_uas=coarse_uas, coarse_vas=coarse_vas)
                    mse, kld, cycle_loss, loss = model.loss_function(p, alpha, beta, fine_pr, mu, log_var,
                                                                     coarse_pr, lambda_kl)
                else:
                    raise ValueError("model {} is not implemented".format(opt.model))

                if loss.item() < float('inf'):
                    loss.backward()
                    # todo refactor this part
                    interval_mse += [mse.item()]
                    interval_kld += [kld.item()]
                    interval_cycle_loss += [cycle_loss.item()]
                    interval_loss += [loss.item()]
                    epoch_mse += mse.item()
                    epoch_kld += kld.item()
                    epoch_cycle_loss += cycle_loss.item()
                    epoch_loss += loss.item()
                    optimizer.step()
                else:
                    print("inf loss")
                iter_time += time.time()-iter_start_time
                iter_data_time += iter_start_time-iter_data_start_time

                if batch_idx % opt.log_interval == 0:
                    viz.print(epoch, batch_idx, sum(interval_mse)/len(interval_mse), sum(interval_kld)/len(interval_kld),
                              sum(interval_cycle_loss)/len(interval_cycle_loss), sum(interval_loss)/len(interval_loss), iter_time,
                              iter_data_time, sum(data['time']).item())
                    iter_data_time = 0
                    iter_time = 0
                    interval_mse = []
                    interval_kld = []
                    interval_cycle_loss = []
                    interval_loss = []
                if batch_idx % opt.plot_interval == 0:
                    img_id += 1
                    image_name = "Epoch{}_Image{}.jpg".format(epoch, img_id)
                    if opt.model == "mse_vae":
                        viz.plot(fine_pr=fine_pr, recon_pr=recon_pr, image_name=image_name)
                    elif opt.model == "gamma_vae":
                        viz.plot(fine_pr=fine_pr, recon_pr=p*alpha*beta, image_name=image_name)
                    else:
                        raise ValueError("model {} is not implemented".format(opt.model))

                if batch_idx % opt.save_latest_interval == 0:
                    save('latest', save_root, opt.gpu_ids, model)
                    print('saved latest epoch after {} iterations'.format(batch_idx))

                # todo add performance evaluation on valitation set periodically

                #if batch_idx % opt.eval_val_loss == 0:
                #    model.eval() # todo set model.train() above
                #    val_climate_dataset.init_epoch() # todo implement val_climate_dataset
                    #todo set all losses zero




                iter_data_start_time = time.time()

            if epoch % opt.save_interval == 0:
                save(epoch, save_root, opt.gpu_ids, model)
        epoch_time = time.time() - epoch_start_time
        viz.print_epoch(epoch=epoch, epoch_mse=epoch_mse,
                        epoch_kld=epoch_kld, epoch_cycle_loss=epoch_cycle_loss,
                        epoch_loss=epoch_loss, epoch_time=epoch_time)


if __name__ == '__main__':
    main()
