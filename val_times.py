import datetime
import torch
import os
from netCDF4 import Dataset

from options.base_options import BaseOptions
from datasets.climate_dataset import ClimateDataset
from utils.upscale import Upscale
from models.gamma_vae import GammaVae


def main():
    # set variables, create directories
    opt = BaseOptions().parse()
    n_samples = opt.n_samples
    device = torch.device("cuda" if len(opt.gpu_ids) > 0 else "cpu")
    upscaler = Upscale(size=48, scale_factor=opt.scale_factor, device=device)
    load_root = os.path.join('checkpoints', opt.name)
    load_epoch = opt.load_epoch if opt.load_epoch >= 0 else 'latest'
    load_name = "epoch_{}.pth".format(load_epoch)
    load_dir = os.path.join(load_root, load_name)
    outdir = os.path.join(opt.results_dir, opt.name, 'times_%s_%s' % (opt.phase, load_epoch))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    climate_data = ClimateDataset(opt=opt, phase=opt.phase)

    # large_cell = 48x48, cell = 40x40, small_cell = 32x32
    cell = 40

    # load the model
    model = GammaVae(opt=opt, device=device).to(device)
    model.load_state_dict(torch.load(load_dir, map_location='cpu'))
    model.eval()


    input_dataset = Dataset(os.path.join(opt.dataroot, 'dataset.nc4'), "r", format="NETCDF4")

    basename = "val.nc4"
    print("start validation")

    # create output file
    output_dataset_path = os.path.join(outdir, basename)
    output_dataset = Dataset(output_dataset_path, "w", format="NETCDF4")
    output_dataset.setncatts({k: input_dataset.getncattr(k) for k in input_dataset.ncattrs()})
    # add own metadata
    output_dataset.creators = "Simon Treu (EDGAN, sitreu@pik-potsdam.de)\n" + output_dataset.creators
    output_dataset.history = datetime.date.today().isoformat() + " Added Downscaled images in python with pix2pix-edgan\n" + output_dataset.history

    output_dataset.createDimension("time", None)
    output_dataset.createDimension("lon", 720)
    output_dataset.createDimension("lat", 120)

    # Copy variables
    for v_name, varin in input_dataset.variables.items():
        outVar = output_dataset.createVariable(v_name, varin.datatype, varin.dimensions)
        # Copy variable attributes
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
    # Create variable for downscaling
    for k in range(n_samples):
        downscaled_pr = output_dataset.createVariable("downscaled_pr_{}".format(k), output_dataset['pr'].datatype,
                                                      output_dataset['pr'].dimensions)
        downscaled_pr.setncatts({k: output_dataset['pr'].getncattr(k) for k in output_dataset['pr'].ncattrs()})
        downscaled_pr.standard_name += '_downscaled'
        downscaled_pr.long_name += '_downscaled'
        downscaled_pr.comment = 'downscaled ' + downscaled_pr.comment

        if opt.model == 'gamma_vae':
            # p
            p = output_dataset.createVariable('p_{}'.format(k), output_dataset['pr'].datatype,
                                              output_dataset['pr'].dimensions)
            p.setncatts({k: output_dataset['pr'].getncattr(k) for k in output_dataset['pr'].ncattrs()})
            p.standard_name += '_p'
            p.long_name += '_p'
            p.comment = 'p ' + p.comment
            # alpha
            alpha = output_dataset.createVariable('alpha_{}'.format(k), output_dataset['pr'].datatype,
                                                  output_dataset['pr'].dimensions)
            alpha.setncatts({k: output_dataset['pr'].getncattr(k) for k in output_dataset['pr'].ncattrs()})
            alpha.standard_name += '_alpha'
            alpha.long_name += '_alpha'
            alpha.comment = 'alpha ' + alpha.comment
            # beta
            beta = output_dataset.createVariable('beta_{}'.format(k), output_dataset['pr'].datatype,
                                                 output_dataset['pr'].dimensions)
            beta.setncatts({k: output_dataset['pr'].getncattr(k) for k in output_dataset['pr'].ncattrs()})
            beta.standard_name += '_beta'
            beta.long_name += '_beta'
            beta.comment = 'beta ' + beta.comment
            # mean_downscaled_pr_{}
            mean_downscaled_pr = output_dataset.createVariable('mean_downscaled_pr_{}'.format(k),
                                                               output_dataset['pr'].datatype,
                                                               output_dataset['pr'].dimensions)
            mean_downscaled_pr.setncatts({k: output_dataset['pr'].getncattr(k) for k in output_dataset['pr'].ncattrs()})
            mean_downscaled_pr.standard_name += '_mean_downscaled_pr'
            mean_downscaled_pr.long_name += '_mean_downscaled_pr'
            mean_downscaled_pr.comment = 'mean_downscaled_pr ' + mean_downscaled_pr.comment

    bilinear_downscaled_pr = output_dataset.createVariable('bilinear_downscaled_pr', output_dataset['pr'].datatype,
                                                           output_dataset['pr'].dimensions)
    bilinear_downscaled_pr.setncatts({k: output_dataset['pr'].getncattr(k) for k in output_dataset['pr'].ncattrs()})
    bilinear_downscaled_pr.standard_name += '_bilinear_downscaled'
    bilinear_downscaled_pr.long_name += '_bilinear_downscaled'
    bilinear_downscaled_pr.comment = 'bilinear_downscaled ' + bilinear_downscaled_pr.comment

    # set variable values
    output_dataset['lat'][:] = input_dataset['lat'][8:-8]
    output_dataset['lon'][:] = input_dataset['lon'][:]
    output_dataset['time'][:] = input_dataset['time'][climate_data.time_list]

    output_dataset['orog'][:] = input_dataset['orog'][8:-8, :]
    output_dataset['pr'][:] = input_dataset['pr'][climate_data.time_list, 8:-8, :]

    output_dataset['uas'][:] = input_dataset['uas'][climate_data.time_list, 8:-8, :]
    output_dataset['vas'][:] = input_dataset['vas'][climate_data.time_list, 8:-8, :]
    output_dataset['psl'][:] = input_dataset['psl'][climate_data.time_list, 8:-8, :]

    for idx_lat in range(3):
        for idx_lon in range(18):

            # lat with index 0 is 34 N.
            large_cell_lats = [i for i in range(idx_lat*40+ 4,(idx_lat+1)*40+12)]
            lats = [i for i in range(idx_lat*40, (idx_lat+1)*40)]
            # longitudes might cross the prime meridian
            large_cell_lons = [i % 720 for i in range(idx_lon*40 - 4,(idx_lon+1)*40 + 4)]
            lons = [i % 720 for i in range(idx_lon*40,(idx_lon+1)*40)]

            pr_tensor = torch.tensor(input_dataset['pr'][climate_data.time_list, large_cell_lats, large_cell_lons],
                                     dtype=torch.float32, device=device).unsqueeze(1)
            orog_tensor = torch.tensor(input_dataset['orog'][large_cell_lats,large_cell_lons],
                                       dtype=torch.float32, device=device).expand(len(climate_data.time_list),1,48,48)
            uas_tensor = torch.tensor(input_dataset['uas'][climate_data.time_list, large_cell_lats, large_cell_lons], dtype=torch.float32, device=device).unsqueeze(1)
            vas_tensor = torch.tensor(input_dataset['vas'][climate_data.time_list, large_cell_lats, large_cell_lons], dtype=torch.float32, device=device).unsqueeze(1)
            psl_tensor = torch.tensor(input_dataset['psl'][climate_data.time_list, large_cell_lats, large_cell_lons], dtype=torch.float32, device=device).unsqueeze(1)

            coarse_pr = upscaler.upscale(pr_tensor)
            coarse_uas = upscaler.upscale(uas_tensor)
            coarse_vas = upscaler.upscale(vas_tensor)
            coarse_psl = upscaler.upscale(psl_tensor)

            for k in range(n_samples):
                with torch.no_grad():
                    recon_pr = model.decode(z=torch.randn(len(climate_data.time_list), opt.nz, 1, 1, device=device),
                                            coarse_pr=coarse_pr, coarse_uas=coarse_uas,
                                            coarse_vas=coarse_vas, orog=orog_tensor,
                                            coarse_psl=coarse_psl)
                if opt.model == "mse_vae":
                    output_dataset['downscaled_pr_{}'.format(k)][:, lats, lons] = recon_pr[:,0,4:-4,4:-4]
                elif opt.model == "gamma_vae":
                    output_dataset['downscaled_pr_{}'.format(k)][:, lats, lons] = \
                        (torch.distributions.bernoulli.Bernoulli(recon_pr['p']).sample() *
                         torch.distributions.gamma.Gamma(recon_pr['alpha'],1/recon_pr['beta']).sample())[:, 0,4:-4,4:-4]

                    output_dataset['mean_downscaled_pr_{}'.format(k)][:,lats,lons] = \
                    torch.nn.Threshold(0.035807043601739474, 0)(recon_pr['p'] * recon_pr['alpha'] * recon_pr['beta'])[:, 0,4:-4,4:-4]

                    output_dataset['p_{}'.format(k)][:,lats,lons] = recon_pr['p'][:, 0,4:-4,4:-4]
                    output_dataset['alpha_{}'.format(k)][:,lats,lons] = recon_pr['alpha'][:, 0,4:-4,4:-4]
                    output_dataset['beta_{}'.format(k)][:,lats,lons] = recon_pr['beta'][:, 0,4:-4,4:-4]
                    #todo don't hardcode threshold
                else:
                    raise ValueError("model {} is not implemented".format(opt.model))

            bilinear_pr = torch.nn.functional.upsample(coarse_pr,scale_factor=opt.scale_factor,
                                                       mode='bilinear',
                                                       align_corners=True)
            output_dataset['bilinear_downscaled_pr'][:, lats, lons] = bilinear_pr[:, 0, 4:-4, 4:-4]
            print('Progress = {:>5.1f} %'.format((idx_lat*3 + idx_lon + 1) * 100 / (3*18)))
    output_dataset.close()
    input_dataset.close()


if __name__=='__main__':
    main()
