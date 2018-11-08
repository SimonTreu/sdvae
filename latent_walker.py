from utils.visualizer import ValidationViz
from options.base_options import BaseOptions
from models.edgan import Edgan

import os.path
import torch


def main():
    opt = BaseOptions().parse()
    device = torch.device("cuda" if len(opt.gpu_ids) > 0 else "cpu")
    load_root = os.path.join('checkpoints', opt.name)
    load_epoch = opt.load_epoch if opt.load_epoch >= 0 else 'latest'
    load_name = "epoch_{}.pth".format(load_epoch)
    load_dir = os.path.join(load_root, load_name)

    # load the model
    edgan_model = Edgan(opt=opt, device=device).to(device)
    edgan_model.load_state_dict(torch.load(load_dir))

    viz = ValidationViz(opt)
    viz.plot_latent_walker(edgan_model, climate_data)

    base_path = os.path.join(save_root, 'epoch_{}_all'.format(opt.load_epoch))
    if not os.path.isfile(os.path.abspath(base_path + '_coarse_pr.pt')):
        all_fine_pr = None
        all_recon_pr = None
        all_coarse = None
        for batch_idx, data in enumerate(climate_data_loader, 0):
            fine_pr = data['fine_pr'].to(device)
            coarse_pr = data['coarse_pr'].to(device)
            cell_area = data['cell_area'].to(device)
            orog = data['orog'].to(device)
            recon_pr = edgan_model.get_picture(coarse_precipitation=coarse_pr,
                                               coarse_ul=data['coarse_ul'].to(device),
                                               coarse_u=data['coarse_u'].to(device),
                                               coarse_ur=data['coarse_ur'].to(device),
                                               coarse_l=data['coarse_l'].to(device),
                                               coarse_r=data['coarse_r'].to(device),
                                               coarse_bl=data['coarse_bl'].to(device),
                                               coarse_b=data['coarse_b'].to(device),
                                               coarse_br=data['coarse_br'].to(device),
                                               orog=orog)
            if not(all_fine_pr is None):
                all_fine_pr = torch.cat((all_fine_pr, fine_pr), 0)
                all_recon_pr = torch.cat((all_recon_pr, recon_pr), 0)
                all_coarse = torch.cat((all_coarse, coarse_pr), 0)
            else:
                all_fine_pr = fine_pr
                all_recon_pr = recon_pr
                all_coarse = coarse_pr
        save_val_data(all_fine_pr, all_coarse, all_recon_pr, base_path)

    val_obj = ValObj(base_path, min=20, max=50)
    val_obj.evaluate_distribution()


if __name__=='__main__':
    main()