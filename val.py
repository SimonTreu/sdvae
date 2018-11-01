import datetime
import torch
import os
from netCDF4 import Dataset

from options.base_options import BaseOptions
from datasets.climate_dataset import ClimateDataset
from utils.upscale import Upscale
from models.edgan import Edgan


def main():
    # set variables, create directories
    opt = BaseOptions().parse()
    n_samples = opt.n_samples
    upscaler = Upscale(size=opt.fine_size+2*opt.scale_factor, scale_factor=opt.scale_factor)
    device = torch.device("cuda" if len(opt.gpu_ids) > 0 else "cpu")
    save_root = os.path.join('checkpoints', opt.name)
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    load_epoch = opt.load_epoch if opt.load_epoch >= 0 else 'latest'
    save_name = "epoch_{}.pth".format(load_epoch)
    save_dir = os.path.join(save_root, save_name)
    outdir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, load_epoch))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    climate_data = ClimateDataset(opt=opt)
    boundary_size = opt.fine_size + 2*opt.scale_factor
    # todo rename, maybe large_cell and small_cell and cell , 48, 32, 40

    # load the model
    edgan_model = Edgan(opt=opt, device=device).to(device)
    edgan_model.load_state_dict(torch.load(save_dir))

    # Iterate val cells and compute #n_samples reconstructions.
    # todo seperate val and test
    index = 0
    # read input file
    input_dataset = Dataset(os.path.join(opt.dataroot, 'dataset.nc4'), "r", format="NETCDF4")
    for idx_lat in range(climate_data.rows):
        for idx_lon in climate_data.test_val_indices[idx_lat][-climate_data.n_val:]:
            # calculate upper left index for cell with boundary values to downscale #todo better formulation
            anchor_lat = idx_lat * climate_data.cell_size  + climate_data.scale_factor
            anchor_lon = idx_lon * climate_data.cell_size
            # select indices for a 48 x 48 box around the 32 x 32 box to be downscaled (with boundary values)
            boundary_lats = [i for i in
                             range(anchor_lat - climate_data.scale_factor,
                                   anchor_lat + climate_data.fine_size + climate_data.scale_factor)]
            # longitudes might cross the prime meridian
            boundary_lons = [i % 720
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
            output_dataset.createDimension("lon", boundary_size)
            output_dataset.createDimension("lat", boundary_size)

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

            bilinear_upscaled_pr = output_dataset.createVariable('bilinear_downscaled_pr', output_dataset['pr'].datatype,
                                                              output_dataset['pr'].dimensions)
            bilinear_upscaled_pr.setncatts({k: output_dataset['pr'].getncattr(k) for k in output_dataset['pr'].ncattrs()})
            bilinear_upscaled_pr.standard_name += '_bilinear_downscaled'
            bilinear_upscaled_pr.long_name += '_bilinear_downscaled'
            bilinear_upscaled_pr.comment = 'bilinear_downscaled ' + bilinear_upscaled_pr.comment
            # set variable values
            output_dataset['lat'][:] = input_dataset['lat'][boundary_lats]
            output_dataset['lon'][:] = input_dataset['lon'][boundary_lons]
            output_dataset['time'][:] = input_dataset['time'][:]
            # crop to 32x32

            output_dataset['orog'][:] = input_dataset['orog'][boundary_lats, boundary_lons]
            output_dataset['pr'][:] = input_dataset['pr'][:, boundary_lats, boundary_lons]
            for k in range(n_samples):
                output_dataset['downscaled_pr_{}'.format(k)][:] = input_dataset['pr'][:, boundary_lats, boundary_lons]

            output_dataset['uas'][:] = input_dataset['uas'][:, boundary_lats, boundary_lons]
            output_dataset['vas'][:] = input_dataset['vas'][:, boundary_lats, boundary_lons]

            # read out the variables similar to construct_datasets.py
            pr = output_dataset['pr'][:]

            uas = output_dataset['uas'][:]
            vas = output_dataset['vas'][:]
            orog = output_dataset['orog'][:]

            times = pr.shape[0]
            for t in range(times):
                pr_tensor = torch.Tensor(pr[t, :, :])
                orog_tensor = torch.Tensor(orog[opt.scale_factor:-opt.scale_factor,
                                           opt.scale_factor:-opt.scale_factor]).unsqueeze(0).unsqueeze(0)
                uas_tensor = torch.Tensor(uas[t, :, :])
                vas_tensor = torch.Tensor(vas[t, :, :])

                coarse_pr = upscaler.upscale(pr_tensor).unsqueeze(0).unsqueeze(0)
                coarse_uas = upscaler.upscale(uas_tensor).unsqueeze(0).unsqueeze(0)
                coarse_vas = upscaler.upscale(vas_tensor).unsqueeze(0).unsqueeze(0)
                for k in range(n_samples):
                    with torch.no_grad():
                        recon_pr = edgan_model.decode(z=torch.randn(1, opt.nz, 1, 1, device=device),
                                                      coarse_pr=coarse_pr,coarse_uas=coarse_uas,
                                                      coarse_vas=coarse_vas, orog=orog_tensor)

                    output_dataset['downscaled_pr_{}'.format(k)][t, opt.scale_factor:-opt.scale_factor, opt.scale_factor:-opt.scale_factor] = recon_pr

                upsample = torch.nn.Upsample(scale_factor=opt.scale_factor, mode='bilinear')
                # todo align_corners=True?
                # todo upsample the complete image and then take only the necessary part
                bilinear_pr = upsample(coarse_pr[:, :, 1:-1, 1:-1])
                output_dataset['bilinear_downscaled_pr'][t, opt.scale_factor:-opt.scale_factor, opt.scale_factor:-opt.scale_factor] = bilinear_pr

            # read out the variables
            # create reanalysis result (croped to 32*32)
            # compute downscaled images
            # save them in result dataset
            output_dataset.close()
            index += 1
        input_dataset.close()

    '''
    # Plot Figure to evaluate latent space
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
    
    '''


if __name__=='__main__':
    main()
