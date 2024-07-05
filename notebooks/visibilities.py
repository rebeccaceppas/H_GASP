import sys

sys.path.append('/Users/rebeccaceppas/Desktop/CHORD/H-GASP')

import observations as obs

output_directory = './vis'
btm_directory = './beams'
map_filepaths = ['map1s.h5', 'map2.h5', 'map3.h5']
map_tags = ['map1s', 'map2', 'map3']

vis = obs.Visibilities(output_directory, btm_directory, map_filepaths, map_tags)

print('changing config')
vis.change_config()
print('done')