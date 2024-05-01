import numpy as np
import h5py
import healpy as hp
from scipy import stats as ss

from FreqState import FreqState
import Generate_HI_Spectra as g
import channelization_functions as cf


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






def sample_profile(fstate, gal_freqs, gal_signal):
    '''
    Samples quasi-continuous (very finely sampled) profile to the desired 
    frequency resolution as given in fstate

    Inputs
    - fstate: <object>
      frequency specifications object from FreqState()
    - gal_freqs: <array> 
      finely sampled frequencies of the ideal profiles
      given from GalaxyCatalog obs_freq property
    - gal_signal: <array>
      temperatures at the finely sampled frequencies of the ideal profiles
      given from GalaxyCatalog T property

    Outputs
    - binned_Ts: <array>
      sampled temperatures corresponding to fstate
    '''
    bin_centres = np.flip(fstate.frequencies)
    bin_edges = bin_centres + fstate.freq_width/2
    bin_edges = np.append(bin_centres[0] - fstate.freq_width/2, bin_edges)
    
    num_freqs = fstate.frequencies.shape[0]
    num_gals = gal_freqs.shape[0]
    
    binned_Ts = np.ones((num_gals, num_freqs))

    for i, f in enumerate(gal_freqs):
        T_, _, _ = ss.binned_statistic(f,
                                        gal_signal[i],
                                        statistic='mean',
                                        bins=bin_edges)

        T_ = np.nan_to_num(T_)
        binned_Ts[i] = np.flip(T_)

    return binned_Ts

def write_map(filename, data, freq, fwidth=None, include_pol=True):
    # Write out the map into an HDF5 file.

    # Make into 3D array
    if data.ndim == 3:
        polmap = np.array(["I", "Q", "U", "V"])
    else:
        if include_pol:
            data2 = np.zeros((data.shape[0], 4, data.shape[1]), dtype=data.dtype)
            data2[:, 0] = data
            data = data2
            polmap = np.array(["I", "Q", "U", "V"])
        else:
            data = data[:, np.newaxis, :]
            polmap = np.array(["I"])

    # Construct frequency index map
    freqmap = np.zeros(len(freq), dtype=[("centre", np.float64), ("width", np.float64)])
    freqmap["centre"][:] = freq
    freqmap["width"][:] = fwidth if fwidth is not None else np.abs(np.diff(freq)[0])

    # Open up file for writing
    with h5py.File(filename, "w") as f:
        f.attrs["__memh5_distributed_file"] = True

        dset = f.create_dataset("map", data=data)
        dt = h5py.special_dtype(vlen=str)
        dset.attrs["axis"] = np.array(["freq", "pol", "pixel"]).astype(dt)
        dset.attrs["__memh5_distributed_dset"] = True

        dset = f.create_dataset("index_map/freq", data=freqmap)
        dset.attrs["__memh5_distributed_dset"] = False
        dset = f.create_dataset("index_map/pol", data=polmap.astype(dt))
        dset.attrs["__meSmh5_distributed_dset"] = False
        dset = f.create_dataset("index_map/pixel", data=np.arange(data.shape[2]))
        dset.attrs["__memh5_distributed_dset"] = False

def make_map(fstate, temp, nside, pol, ra, dec, write=False, filename=None, new=True, existing_map=None):
    """
    Creates the galaxy map

    Parameters:
    - fstate <object>: object created from FreqState including start and end of sampled frequencies and number of bins
    - temp <array>: binned_T from sample_profile function
    - nside <int>: not sure how to explain
    - pol <str>: full for all polarizations, can also choose I, Q, V, U only (I think)
    - ra <float>: position in the sky in degrees (Right Ascension)
    - dec <float>: position in the sky in degrees (Declination)
    - write <bool>: tells the function to save the file or just return the map
    - filename <str>: name of file to save if write=True
    - new <bool>: tells function if need to create a new map from start or will be providing existing map
    - existing_map <h5 map>: previously created map onto which to add more stuff

    Returns:
    map_ <h5 map>: map with galaxy profile injected into
    """
    nfreq = len(fstate.frequencies)
    npol = 4 if pol == "full" else 1

    if new:
        map_ = np.zeros((nfreq, npol, 12 * nside**2), dtype=np.float64)

    else:  
        map_ = np.copy(existing_map)
    
    for i in range(nfreq):
        map_[i, 0, hp.ang2pix(nside, ra, dec, lonlat=True)] += temp[i]  # removed the flip because added it in the sampling function

    if write:
        write_map(filename, map_, fstate.frequencies, fstate.freq_width, include_pol=True)

    return map_

def map_catalog(fstate, temp, nside, pol, ras, decs, filename=None, write=True):
    """
    Creates a map containing the given HI galaxy catalog specifications
    New function due to the fact that cannot seem to install healpy on cedar
    
    Parameters:
    - fstate <object>: object created from FreqState including start and end of sampled frequencies and number of bins
    - temp <ndarray>: binned_Ts from sample_profile function containing all of the binned profiles to inject
    - nside <int>: defines the resolution of the map
    - pol <str>: full for all polarizations, can also choose I, Q, V, U only (I think)
    - pix <int>: pre-calculated pixel
    - write <bool>: tells the function to save the file or just return the map
    - filename <str>: name of file to save if write=True
    - new <bool>: tells function if need to create a new map from start or will be providing existing map
    - existing_map <h5 map>: previously created map onto which to add more stuff

    Returns:
    map_ <h5 map>: map with galaxy profile injected into
    """

    nfreq = len(fstate.frequencies)
    npol = 4 if pol =='full' else 1
    ngal = len(ras)

    map_ = np.zeros((nfreq, npol, 12* nside**2), dtype=np.float64)

    for i in range(ngal):
        ra = ras[i]
        dec = decs[i]
        T = temp[i]
        for j in range(nfreq):
            map_[j, 0, hp.ang2pix(nside, ra, dec, lonlat=True)] += T[j]

    if write:
        write_map(filename, map_, fstate.frequencies, fstate.freq_width, include_pol=True)

    return map_
