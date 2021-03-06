import datetime
import torch
import os
from netCDF4 import Dataset

from options.base_options import BaseOptions
from datasets.climate_dataset import ClimateDataset
from utils.upscale import Upscale
from models.sdvae import SDVAE


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
    outdir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, load_epoch))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    climate_data = ClimateDataset(opt=opt, phase=opt.phase)

    # large_cell = 48x48, cell = 40x40, small_cell = 32x32
    large_cell = opt.fine_size + 2*opt.scale_factor

    # load the model
    model = SDVAE(opt=opt, device=device).to(device)
    model.load_state_dict(torch.load(load_dir, map_location='cpu'))
    model.eval()

    # Iterate val cells and compute #n_samples reconstructions.
    index = 0
    # read input file
    input_dataset = Dataset(os.path.join(opt.dataroot, 'dataset.nc4'), "r", format="NETCDF4")
    for idx_lat in range(climate_data.rows):
        for idx_lon in climate_data.lat_lon_list[idx_lat]:
            # calculate upper left index for cell with boundary values to downscale
            anchor_lat = idx_lat * climate_data.cell_size  + climate_data.scale_factor
            anchor_lon = idx_lon * climate_data.cell_size
            # select indices for a 48 x 48 box around the 32 x 32 box to be downscaled (with boundary values)
            large_cell_lats = [i for i in
                               range(anchor_lat - climate_data.scale_factor,
                                     anchor_lat + climate_data.fine_size + climate_data.scale_factor)]
            # longitudes might cross the prime meridian
            large_cell_lons = [i % 720
                             for i in
                             range(anchor_lon - climate_data.scale_factor,
                                   anchor_lon + climate_data.fine_size + climate_data.scale_factor)]
            # create output path
            basename = "val.lat{}_lon{}.nc4".format(anchor_lat, anchor_lon)
            print("test file nr. {} name: {}".format(index, basename))

            # create output file
            output_dataset_path = os.path.join(outdir, basename)
            output_dataset = Dataset(output_dataset_path, "w", format="NETCDF4")
            output_dataset.setncatts({k: input_dataset.getncattr(k) for k in input_dataset.ncattrs()})
            # add own metadata
            output_dataset.creators = "Simon Treu (EDGAN, sitreu@pik-potsdam.de)\n" + output_dataset.creators
            output_dataset.history = datetime.date.today().isoformat() + " Added Downscaled images in python with pix2pix-edgan\n" + output_dataset.history

            output_dataset.createDimension("time", None)
            output_dataset.createDimension("lon", large_cell)
            output_dataset.createDimension("lat", large_cell)

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
                    mean_downscaled_pr = output_dataset.createVariable('mean_downscaled_pr_{}'.format(k), output_dataset['pr'].datatype,
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
            output_dataset['lat'][:] = input_dataset['lat'][large_cell_lats]
            output_dataset['lon'][:] = input_dataset['lon'][large_cell_lons]
            output_dataset['time'][:] = input_dataset['time'][:]

            output_dataset['orog'][:] = input_dataset['orog'][large_cell_lats, large_cell_lons]
            output_dataset['pr'][:] = input_dataset['pr'][:, large_cell_lats, large_cell_lons]
            for k in range(n_samples):
                output_dataset['downscaled_pr_{}'.format(k)][:] = input_dataset['pr'][:, large_cell_lats, large_cell_lons]
            output_dataset['bilinear_downscaled_pr'][:] = input_dataset['pr'][:, large_cell_lats, large_cell_lons]

            output_dataset['uas'][:] = input_dataset['uas'][:, large_cell_lats, large_cell_lons]
            output_dataset['vas'][:] = input_dataset['vas'][:, large_cell_lats, large_cell_lons]
            output_dataset['psl'][:] = input_dataset['psl'][:, large_cell_lats, large_cell_lons]

            # read out the variables similar to construct_datasets.py
            pr = output_dataset['pr'][:]

            uas = output_dataset['uas'][:]
            vas = output_dataset['vas'][:]
            psl = output_dataset['psl'][:]
            orog = output_dataset['orog'][:]

            times = pr.shape[0]
            for t in range(times):
                pr_tensor = torch.tensor(pr[t, :, :], dtype=torch.float32, device=device)
                orog_tensor = torch.tensor(orog[:,:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                uas_tensor = torch.tensor(uas[t, :, :], dtype=torch.float32, device=device)
                vas_tensor = torch.tensor(vas[t, :, :], dtype=torch.float32, device=device)
                psl_tensor = torch.tensor(psl[t, :, :], dtype=torch.float32, device=device)

                coarse_pr = upscaler.upscale(pr_tensor).unsqueeze(0).unsqueeze(0)
                coarse_uas = upscaler.upscale(uas_tensor).unsqueeze(0).unsqueeze(0)
                coarse_vas = upscaler.upscale(vas_tensor).unsqueeze(0).unsqueeze(0)
                coarse_psl = upscaler.upscale(psl_tensor).unsqueeze(0).unsqueeze(0)

                for k in range(n_samples):
                    with torch.no_grad():
                        recon_pr = model.decode(z=torch.randn(1, opt.nz, 1, 1, device=device),
                                                coarse_pr=coarse_pr, coarse_uas=coarse_uas,
                                                coarse_vas=coarse_vas, orog=orog_tensor,
                                                coarse_psl=coarse_psl)
                    if opt.model == "mse_vae":
                        output_dataset['downscaled_pr_{}'.format(k)][t, :, :] = recon_pr
                    elif opt.model == "gamma_vae":
                        output_dataset['downscaled_pr_{}'.format(k)][t, :, :] = \
                            (torch.distributions.bernoulli.Bernoulli(recon_pr['p']).sample() *
                             torch.distributions.gamma.Gamma(recon_pr['alpha'],1/recon_pr['beta']).sample())[0, 0,:,:]

                        output_dataset['mean_downscaled_pr_{}'.format(k)][t,:,:] = \
                        torch.nn.Threshold(0.035807043601739474, 0)(recon_pr['p'] * recon_pr['alpha'] * recon_pr['beta'])

                        output_dataset['p_{}'.format(k)][t,:,:] = recon_pr['p']
                        output_dataset['alpha_{}'.format(k)][t,:,:] = recon_pr['alpha']
                        output_dataset['beta_{}'.format(k)][ t,:,:] = recon_pr['beta']
                        #todo don't hardcode threshold
                    else:
                        raise ValueError("model {} is not implemented".format(opt.model))

                bilinear_pr = torch.nn.functional.upsample(coarse_pr,scale_factor=opt.scale_factor,
                                                           mode='bilinear',
                                                           align_corners=True)
                output_dataset['bilinear_downscaled_pr'][t, :, :] = bilinear_pr[0, 0, :, :]

            # read out the variables
            # create reanalysis result (croped to 32*32)
            # compute downscaled images
            # save them in result dataset
            output_dataset.close()
            index += 1
    input_dataset.close()


if __name__=='__main__':
    main()
