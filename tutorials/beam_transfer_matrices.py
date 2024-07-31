import sys

# change this to the path where you've cloned the pipeline
H_GASP_path = '/home/rebeccac/scratch/H_GASP'

sys.path.append(H_GASP_path)

from H_GASP import observations as obs
from H_GASP import frequencies as fr

fmax = 1420
fmin = 1419
U = 2

fstate, f_start, f_end, nfreq = fr.get_frequencies(fmax, fmin, U)

output_directory = './beams'

CHORDdec_pointing = 10
n_dishes_ew = 2
n_dishes_ns = 1

btm = obs.BeamTransferMatrices(f_start, f_end, nfreq, output_directory, H_GASP_path,
                               CHORDdec_pointing, n_dishes_ew=n_dishes_ew,
                               n_dishes_ns=n_dishes_ns)


btm.get_beam_transfer_matrices()
