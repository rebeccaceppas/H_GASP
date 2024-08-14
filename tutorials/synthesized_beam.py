H_GASP_path = '/home/rebeccac/scratch/H_GASP'

import sys

sys.path.append(H_GASP_path)

from H_GASP import input_maps as im
from H_GASP import frequencies as fr
from H_GASP import observations as obs

'''--------Set up---------'''
f_start = 1420
f_end = 1419
nfreq = 2  # this is the minimum nfreq

fstate = fr.FreqState()
fstate.freq = (f_start, f_end, nfreq)

nside = 16  # for the map resolution

# position of the source
ra = 180
dec = 45

# where you want your synthesized beam products to live
output_directory = './synth_beam/'
output_filename = 'input_synth_beam_RA180_DEC45.h5'

T_brightness = 1.0


'''--------Input map for the synthesized beam---------'''
synth_beam = im.SynthesizedBeam(f_start, f_end, nfreq)

synth_beam.get_map(nside, ra, dec, 
                   output_directory, output_filename=output_filename,
                   T_brightness=T_brightness)

'''--------Simulate beam transfer matrices----------'''

btm_directory = output_directory

CHORDdec_pointing = 10  # declination in degrees
n_dishes_ew = 2
n_dishes_ns = 1
spacing_ew = 6.3
spacing_ns = 8.5
dish_diameter = 6.0

btm = obs.BeamTransferMatrices(f_start, f_end, nfreq, btm_directory, H_GASP_path,
                               CHORDdec_pointing,
                               n_dishes_ew, n_dishes_ns, spacing_ew, spacing_ns,
                               dish_diameter)

btm.get_beam_transfer_matrices()

'''-----------Visibilities and dirty map (synthesized beam)-----------'''
maps_tag = 'synth_beam'  # a single tag to remember which components were in the input maps
map_filepaths = [output_directory+output_filename]

vis = obs.Visibilities(output_directory, btm_directory, H_GASP_path, maps_tag, map_filepaths)
visdata = vis.get_visibilities()

dirty_map_filename = 'synthesized_beam_RA180_DEC45.h5'

dm = obs.DirtyMap(visdata,                         
                  output_directory,
                  dirty_map_filename,   
                  fstate, nside, btm_directory)

dm.get_dirty_map()

