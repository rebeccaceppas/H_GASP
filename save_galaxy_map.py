import numpy as np
import h5py
import healpy as hp
from scipy import stats as ss

def sample_profile(fstate, gal_freqs, gal_signal, catalog=False):

    """samples the profile bases on frequencies of beam/map
    
    Parameters:
    - fstate <object>= FreqState()
    - gal_freqs <array>: array containing the finely sampled frequencies of the ideal profiles
                         given from GalaxyProfile obs_freq method
    - gal_signal <array>: array containing the temperatures at the finely sampled frequencies of the ideal profiles
                          given from GalaxyProfile .T method

    Returns:
    - binned_T <array>: temperature signal at each of the binner frequency intervals
     """

    bin_centres = np.flip(fstate.frequencies)
    bin_edges = bin_centres + fstate.freq_width/2
    bin_edges = np.append(bin_centres[0] - fstate.freq_width/2, bin_edges)
    
    num_freqs = fstate.frequencies.shape[0]
    num_gals = gal_freqs.shape[0]
    
    binned_Ts = np.ones((num_gals, num_freqs))
    if catalog:
        for i, f in enumerate(gal_freqs):
            T_, _, _ = ss.binned_statistic(f,
                                           gal_signal[i],
                                           statistic='mean',
                                           bins=bin_edges)

            T_ = np.nan_to_num(T_)

            binned_Ts[i] = np.flip(T_)


    else:
        binned_Ts, _, _ = ss.binned_statistic(gal_freqs, 
                                         gal_signal,
                                         statistic='mean',
                                         bins=bin_edges)
    
        binned_Ts = np.flip(np.nan_to_num(binned_Ts))

    # flipping the array so that it shows the variation as the frequency decreases 
    # matched what we do in the map making, make sure I'm not flipping there again

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

def cedar_map(fstate, temp, nside, pol, pix, write=False, filename=None, new=True, existing_map=None):
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
        map_[i, 0, pix] += temp[i] 

    if write:
        write_map(filename, map_, fstate.frequencies, fstate.freq_width, include_pol=True)

    return map_