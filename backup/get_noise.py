from noise import NormalizedNoise, GaussianNoise, get_manager, get_sstream
import yaml
import numpy as np
from drift.core import manager
from draco.analysis import mapmaker, transform, flagging
from save_galaxy_map import write_map
from FreqState import FreqState

yaml_input_file = open("inputs.yaml")
input = yaml.safe_load(yaml_input_file)
output_folder = input['process']['output_folder']
yaml_input_file.close()

yaml_file = open(output_folder+'/outputs.yaml')
output = yaml.safe_load(yaml_file)
tsys = output['telescope']['tsys']
ndays = output['telescope']['ndays']
nside = output['telescope']['nside']
f_start = output['fstate']['f_start']
f_end = output['fstate']['f_end']
nfreq = output['fstate']['nfreq']
yaml_file.close()

set_weights = True
add_noise = True

dict_stream = {'recv_temp': tsys, 
               'ndays': ndays}

dict_map = {'nside': nside}

dict_mask = {'auto_correlations': False}

print(ndays)

norm = np.load(output_folder+'/norm.npy')
manager = get_manager(output_folder)
data = get_sstream(output_folder)

fstate = FreqState()
fstate.freq = (f_start, f_end, nfreq)

#### NORMALIZED NOISE

'''getting noisy visibilities'''
noisy = NormalizedNoise()
noisy.setup(manager)
noisy.read_config(dict_stream)
noisy_data = noisy.process(data, norm)

'''getting M-modes'''
mmodes = transform.MModeTransform()
mmodes.setup(manager)
Mmodes = mmodes.process(noisy_data)

'''masking auto correlations'''
mmodes_masked = flagging.MaskMModeData()
mmodes_masked.read_config(dict_mask)
Mmodes_masked = mmodes_masked.process(Mmodes)

'''making dirty map'''
dm = mapmaker.DirtyMapMaker()
dm.read_config(dict_map)
dm.setup(manager)
m = dm.process(Mmodes_masked)
filename = output_folder+'/dirty_map_norm_{}.h5'.format(ndays)
map_ = m['map'][:]

write_map(filename, map_, fstate.frequencies, fstate.freq_width, include_pol=True)

#### GAUSSIAN NOISE

'''getting noisy visibilities'''
noisy_gauss = GaussianNoise()
noisy_gauss.setup(manager)
noisy_gauss.read_config(dict_stream)
noisy_data_gauss = noisy_gauss.process(data)

'''getting M-modes'''
mmodes_gauss = transform.MModeTransform()
mmodes_gauss.setup(manager)
Mmodes_gauss = mmodes_gauss.process(noisy_data_gauss)

'''masking auto correlations'''
mmodes_gauss_masked = flagging.MaskMModeData()
mmodes_gauss.read_config(dict_mask)
Mmodes_gauss_masked = mmodes_gauss_masked.process(Mmodes_gauss)

'''making dirty map'''
dm_gauss = mapmaker.DirtyMapMaker()
dm_gauss.read_config(dict_map)
dm_gauss.setup(manager)
m_gauss = dm_gauss.process(Mmodes_gauss_masked)
filename = output_folder+'/dirty_map_gauss_{}.h5'.format(ndays)
map_gauss = m_gauss['map'][:]

write_map(filename, map_gauss, fstate.frequencies, fstate.freq_width, include_pol=True)
