import sys

sys.path.append('/home/rebeccac/scratch/H-GASP')

import observations as obs
import frequencies as fr

fmax = 1420
fmin = 1419
U = 4

print('setting up frequencies')
fstate, f_start, f_end, nfreq = fr.get_frequencies(fmax, fmin, U)

output_directory = './beams'

CHORDdec_pointing = 10
n_dishes_ew = 2
n_dishes_ns = 2

print('initializing object')
btm = obs.BeamTransferMatrices(f_start, f_end, nfreq, output_directory,
                               CHORDdec_pointing, n_dishes_ew=n_dishes_ew,
                               n_dishes_ns=n_dishes_ns)



btm.change_config()
#print('getting matrices')
#btm.get_beam_transfer_matrices()

print('done')