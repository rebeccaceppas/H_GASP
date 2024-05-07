import numpy as np
from savetools import make_map, map_catalog
from FreqState import FreqState
import channelization_functions as cf
import Generate_HI_Spectra as g

def inject_synthesized_beam(ra, dec, 
                            f_start=1418, f_end=1417, nfreq=2, 
                            nside=512, filename=None):

    '''Injects a source in a map for simulating synthesized beams with dirty map maker

    Inputs:
    ------
    - ra: <float>
      right ascension in degrees
    - dec: <float>
      declination in degrees

    Outputs:
    -------
    '''

    fstate = FreqState()
    fstate.freq = (f_start, f_end, nfreq)

    binned_temps = np.ones(nfreq)

    pol='full'

    if filename is None:
        filename = 'input_RA{0:.2f}_DEC{0:.2f}.h5'.format(ra, dec)

    map_input = make_map(fstate,
                         binned_temps,
                         nside, pol,
                         ra, dec,
                         write=True,
                         filename=filename,
                         new=True, existing_map=None)

    return map_input
    

def inject_ngals(R_filepath, 
                 norm_filepath, 
                 filename,
                 ngals='all',
                 save_gal_info=False, 
                 f_start=1420.276874203762, 
                 f_end=1369.3410603548411, 
                 nfreq=1392, 
                 U=16, 
                 nside=512, 
                 catalogue_filepath='/home/rebeccac/scratch/thesis/input_maps/ConstrainSim_dec45.txt'):
    
    '''Inject ngals out of the catalog
    This is for the 2D matched filted to test with multiple Ngal
    The input map will also be used for CLEAN
    
    Note: the maximum for the ALFALFA constrained catalog is ~3,500
    Note2: the catalogs will be upchannelized

    Inputs
    ------
    - ngals: <int> or <str: "all">
    '''

    fstate = FreqState()
    fstate.freq = (f_start, f_end, nfreq)

    fine_freqs = cf.get_fine_freqs(fstate.frequencies)
    
    # Read Catalog:
    Catalog = np.loadtxt(catalogue_filepath)

    # Galaxy parameters:
    MHI = Catalog[0]      # HI Mass - SolMass
    VHI = Catalog[1]      # HI Velocity - km/s
    i = Catalog[2]        # inclination - radians
    D = Catalog[3]        # Distance - Mpc
    z = Catalog[5]        # Redshift 
    ra = Catalog[6]       # Right Ascension - Degrees
    dec = Catalog[7]      # Declination - Degrees

    # Busy function parameters:
    a = Catalog[8]        # Controls peak 
    b1 = Catalog[9]       # Controls height of one peak in double-peak profile
    b2 = Catalog[10]      # Controls height of other peak in double-peak profile
    c = Catalog[11]       # Controls depth of trough

    ## Generate all Spectra from points in catalog into one array V velocity and S flux:
    sample_size = len(Catalog[0])
    if isinstance(ngals, int):
        galaxies_sample = np.random.choice(range(sample_size), ngals, replace=False)
        
    else:
        galaxies_sample = range(sample_size)

    if save_gal_info:
        np.save('{}_galaxy_sample.npy'.format(filename), galaxies_sample)


    # generating their spectra
    V = []; S = []
    for j in galaxies_sample:
        try_M, v, s, w, w_, _, _, _, _ = g.Generate_Spectra(MHI[j], VHI[j], 
                                                            i[j], D[j], 
                                                            a=a[j], b1=b1[j], 
                                                            b2=b2[j], c=c[j])
    V.append(v)
    S.append(s)

    profiles = cf.get_resampled_profiles(V, S, z, fine_freqs)

    # upchannelizing
    heights = cf.upchannelize(profiles, U, R_filepath, norm_filepath)

    pol = 'full'
    map_catalog(fstate, np.flip(heights, axis=1), nside, pol, ra, dec, filename='{}_input_map.h5'.format(filename), write=True)

    return 0



