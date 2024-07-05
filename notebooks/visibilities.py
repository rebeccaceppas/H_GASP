import sys

sys.path.append('/home/rebeccac/scratch/H-GASP/')

import observations as obs
import input_maps as im


# make an input map
f_start = 1420.1488028327337
f_end = 1418.3923954586332
nfreq = 6
#ra = 180  # input map
ra = 10  # map2.h5
dec = 10
nside = 16
output_directory = './vis/'
output_filename = 'input_map.h5'
ngals = 1

#hi_gal = im.SynthesizedBeam(f_start, f_end, nfreq)

#hi_gal.get_map(nside, ra, dec, output_directory, 'map2.h5')

# get visibilities

output_directory = './vis'
btm_directory = './beams'
map_filepaths = [output_directory+'/'+output_filename,
                output_directory+'/'+ 'map2.h5']
maps_tag = 'one_gal'

vis = obs.Visibilities(output_directory, btm_directory, maps_tag, map_filepaths)

print('simulating visibilities')
vis.get_visibilities()
print('done')