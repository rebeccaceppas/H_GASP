## this is an example of how to run every step of the pipeline

## first need to make sure the modules are loaded
'''
module use /project/rrg-kmsmith/shared/chord_env/modules/modulefiles/
module load chord/chord_pipeline/2023.06
'''

import sys

# change this to the path where you've cloned the pipeline
H_GASP_path = '/home/rebeccac/scratch/H_GASP'

sys.path.append(H_GASP_path)

from H_GASP import input_maps as im
from H_GASP import observations as obs
from H_GASP import frequencies as fr

####### setting up the basic parameters for the run #######
fmax = 1420
fmin = 1415
nfreq  = 40  # this is the frequency of the map creation
             # if you are going to up-channelize the map, you can keep this
             # large enough so the shapes of the profiles are not lost pre upchan
nside = 128
output_directory = '/home/rebeccac/scratch/H_GASP/tutorials/full_run/'

U = 2

compute_R = True  # whether you need to compute R and norm for up-channelization
                  # if they have already been compute this script will just straight
                  # to performing the up-channelization

CHORDdec_pointing = 10  # declination in degrees
n_dishes_ew = 2
n_dishes_ns = 1
spacing_ew = 6.3
spacing_ns = 8.5
dish_diameter = 6.0

# for adding noise
ndays = 100

###### creating input maps: one with all galaxies in the galaxy catalog and one with cora-makesky foregrounds ######
catalog_filepath = '/home/rebeccac/scratch/H_GASP/resources/HIMF_dec45_VolLim_10000.txt'
HIgals_filename = 'HI_gals.h5'
hi_gals = im.HIGalaxies(catalog_filepath, fmax, fmin, nfreq)
hi_gals.get_map(nside, output_directory+'/'+HIgals_filename)


components = ['foreground', '21cm']
fg = im.Foregrounds(fmax, fmin, nfreq, nside, output_directory)
fg.get_maps(components)

############## up-channelizing the maps ################

# list of all the maps to up-channelize
map_paths = [output_directory+HIgals_filename,
             output_directory+'foregrounds_all.h5']

upchan_filename = 'upchan_map.h5'
R_filename = 'R_{}_{}_{}.npy'.format(fmax, fmin, U)
norm_filename = 'norm_{}_{}_{}.npy'.format(fmax, fmin, U)
freq_matrix_filename = 'freqs_matrix_{}_{}_{}.npy'.format(fmax, fmin, U)

upchan = obs.Upchannelization(U,
                              fmax,
                              fmin,
                              output_directory,
                              upchan_filename,
                              R_filename,
                              norm_filename,
                              freq_matrix_filename)
if compute_R:
    upchan.get_R_norm()

f_start, f_end, nfreqU = upchan.upchannelize(map_paths=map_paths)

fstate = fr.FreqState()
fstate.freq = (f_start, f_end, nfreqU)

############# computing beam transfer matrices #####################
# choosing the save the matrices in the output directory but can decide on something else
btm_directory = output_directory

btm = obs.BeamTransferMatrices(f_start, f_end, nfreqU, output_directory, H_GASP_path,
                               CHORDdec_pointing,
                               n_dishes_ew, n_dishes_ns, spacing_ew, spacing_ns,
                               dish_diameter, ndays=ndays)

btm.get_beam_transfer_matrices()


############# getting visibilities ##################
maps_tag = 'HIcatalog_foregrounds'  # a single tag to remember which components were in the input maps
map_filepaths = [output_directory+upchan_filename]

vis = obs.Visibilities(output_directory, btm_directory, H_GASP_path, maps_tag, map_filepaths)
visdata = vis.get_visibilities()

############# adding noise and calibration errors ##############

real_vis_obs = obs.RealisticVisibilities(ndays, btm_directory, output_directory, maps_tag)    
noisy_visdata = real_vis_obs.add_noise(upchannelized=True,
                                  norm_filepath=output_directory+norm_filename)

############# getting dirty map #################

dirty_map_filename = 'dirty_map.h5'

dm = obs.DirtyMap(noisy_visdata,                         # if you want the dirty map of the noiseless visibilities
                  output_directory+dirty_map_filename,   # pass visdata as the argument instead
                  fstate, nside, btm_directory)

dm.get_dirty_map()

######## DONE ###########