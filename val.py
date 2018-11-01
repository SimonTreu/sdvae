from datasets.climate_dataset import ClimateDataset
from torch.utils.data import DataLoader
from models.edgan import Edgan
from utils.visualizer import ValidationViz
from netCDF4 import Dataset
import datetime
from utils.upscale import Upscale

import torch
import os
from options.base_options import BaseOptions
from utils.validate_distribution import ValObj, save_val_data
opt = BaseOptions().parse()

n_samples = opt.n_samples
upscaler = Upscale(size=opt.fine_size+2*opt.scale_factor, scale_factor=opt.scale_factor)
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
edgan_model = Edgan(opt=opt, device=device).to(device)

save_name = "epoch_{}.pth".format(opt.load_epoch)
save_dir = os.path.join(save_root, save_name)
edgan_model.load_state_dict(torch.load(save_dir))

# Iterate the netcdf files and plot reconstructed images.
nc4_sample_paths = [os.path.join(opt.dataroot, 'test.0.12.nc4')] # todo make it properly

outdir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.load_epoch))
if not os.path.exists(outdir):
    os.makedirs(outdir)
index = 0
for input_dataset_path in nc4_sample_paths:
    # create output path
    basename = os.path.basename(input_dataset_path)
    outpath = os.path.join(outdir, basename)
    # print
    print("test file nr. {} name: {}".format(index, basename))
    index += 1
    # read input file
    input_dataset = Dataset(input_dataset_path, "r", format="NETCDF4")
    # create output file
    output_dataset_path = os.path.join(outdir, basename)
    output_dataset = Dataset(output_dataset_path, "w", format="NETCDF4")
    output_dataset.setncatts({k: input_dataset.getncattr(k) for k in input_dataset.ncattrs()})
    # add own metadata
    output_dataset.creators = "Simon Treu (EDGAN, sitreu@pik-potsdam.de)\n" + output_dataset.creators
    output_dataset.history = datetime.date.today().isoformat() + " Added Downscaled images in python with pix2pix-edgan\n" + output_dataset.history

    output_dataset.createDimension("time", None)
    output_dataset.createDimension("lon", opt.fine_size+2*opt.scale_factor)
    output_dataset.createDimension("lat", opt.fine_size+2*opt.scale_factor)

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
    for var in ['lat', 'lon']:
        output_dataset[var][:] = input_dataset[var][:]
    output_dataset['time'][:] = input_dataset['time'][:]
    # crop to 32x32

    output_dataset['orog'][:] = input_dataset['orog'][:]
    output_dataset['pr'][:] = input_dataset['pr'][:]
    for k in range(n_samples):
        output_dataset['downscaled_pr_{}'.format(k)][:] = input_dataset['pr'][:]

    output_dataset['uas'][:] = input_dataset['uas'][:]
    output_dataset['vas'][:] = input_dataset['vas'][:]

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
