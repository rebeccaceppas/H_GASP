import numpy as np
import numexpr as ne 
import matplotlib.pyplot as plt
from unit_converter import GalaxyCatalog 
import Generate_HI_Spectra as g
import h5py
from FreqState import FreqState
from save_galaxy_map import write_map, map_catalog
from scipy import interpolate

def window(index, M, length): # window function
    '''(array, int, int) -> (array)
    Passes an array of summation indices through a sinc-hanning window function'''
    return (np.cos(np.pi * (index-M*length/2)/(M*length-1)))**2 * np.sinc((index-M*length/2)/length)

def exponential_chan(s, mtx, N): 
    '''(array, array, int) -> array
    Passes an array (matrix) which needs to be modified and exponentiated'''

    # reshaping (coarse chans, nfreq) -> (coarse chans, nfreq, 1)
    mtx = np.reshape(mtx, (mtx.shape[0], mtx.shape[1], 1))

    # matrix multiplication, shapes = (coarse chans, nfreqs, 1) x (1, M*N) = (coarse chans, nfreqs, M*N)
    v = np.matmul(mtx, s)

    exponent = -2j * np.pi * v / N
    return ne.evaluate("exp(exponent)")

def weight_chan(f, M, N): 
    '''(array, int, int) -> array
    Takes an array containing (c-f) entries and passes it through a first-round PFB''' 

    # creating an array containing the indices over which we need to sum the DFT over 
    # shape = (1, M*N)
    j = np.reshape(np.arange(M*N), (1, M*N)) 

    # getting the window function and exponential bit of the DFT to be multiplied together
    # result has shape (coarse chans, nfreq, M*N)
    summation = window(j, M, N) * exponential_chan(j, f, N)

    # collapsing so it has shape (coarse chans, nfreq) again
    return np.sum(summation, axis = 2)

def exponential_upchan(B, k):
    '''(array, array, int) -> array
    Passes an array (matrix) which needs to be modified and exponentiated'''

    # reshaping (U, nfreq) -> (U, nfreq, 1)
    B = np.reshape(B, (B.shape[0], B.shape[1], 1))

    # matrix multiplication, shapes = (U, nfreqs, 1) x (1, M*U) = (coarse chans, nfreqs, M*U)
    v = np.matmul(B, k)

    exponent = np.pi * 1j * v
    return ne.evaluate("exp(exponent)")

def weight_upchan(B, M, U):
    '''(array, int, int) -> array
    Takes an array containing ((U-1)/U - 2u/U + 2f) entries and passes it through first-round PFB''' 

    # creating an array containing the indices over which we need to sum the DFT over, shape = (1, M*U)
    k = np.reshape(np.arange(M*U), (1, M*U))

    # getting window function and exponential bit of the DFT which need to be multiplied together
    # result has shape (coarse chans, nfreq, M*U)
    summation = window(k, M, U) * exponential_upchan(B, k)

    # summation collapses shape back down to shape = (U, nfreq)
    return np.sum(summation, axis = 2) 

def response_mtx(c, f, M, N, U):
    '''Creating a matrix which can be multiplied against input frequencies to give 
       response on channels
    Inputs:
        f = np.ndarray() of shape (# of frequencies, 1),
            large number of frequencies to simulate continuous 'real' spectrum
        c = np.ndarray() of shape (1, # of coarse channels), channels we are observing on
        M = int, number of taps
        N = int, length of each block
    Outputs: Matrix of size (number of fine channels x number of frequencies / length of profile) 
              which can be multiplied against a galaxy profile/spectrum to yield the response on 
              each fine channel to the profile as a whole
    '''
    # creating coarse channelization matrix, size = (number of coarse chans x nfreq)
    # where each entry is (c-f) of the relevant channel and frequency
    submtx_chan = np.tile(c, [f.shape[0], 1]).T - f[:,0]

    # passing each element in the matrix through the coarse channelization algorithm
    submtx_chan = weight_chan(submtx_chan, M, N)

    # reshaping the resulting matrix so that we get U identical rows per coarse channel
    mtx_chan = np.repeat(submtx_chan, U, axis = 0)  

    # ----------
    # creating fine upchannelization matrix, size = (U x nfreq)
    submtx_upchan = np.tile(np.arange(U), [f.shape[0], 1]).T

    # making it so that every entry corresponds to the expression needed in the exponential term 
    # of the upchannelization weight function
    submtx_upchan = (U-1) / U - 2*submtx_upchan / U + 2*f[:,0]

    # passing through upchannelization algorithm
    submtx_upchan = weight_upchan(submtx_upchan, M, U)  

    # reshaping so that we get repeating blocks from u = 1...U for each coarse channel
    mtx_upchan = np.tile(submtx_upchan, (len(c[0]), 1)) # tiling resulting matrix to correct shape

    # returning the combined response matrix where coarse and fine channelization weights are multiplied
    return np.multiply(mtx_chan, mtx_upchan)

def freq_unit_strip(f):
    '''strips quantities in frequency-space (MHz) to become unitless'''
    return (f - 300) * 4096 * 0.417 * 0.001

def freq_unit_add(f_bar):
    '''adds frequency units (MHz) to unitless quantities'''
    return f_bar / (4096 * 0.417 * 0.001) + 300

def get_fine_freqs(observing_freqs):
    fmax = np.max(observing_freqs)+2
    fmin = np.min(observing_freqs)-2
    dc = observing_freqs[1] - observing_freqs[0]  # getting a negative dc 
    return np.arange(fmax, fmin, dc / 3) # making the frequency resolution = 1/3 dc 

def get_chans(min_freq, max_freq):
    '''(int/float, int/float) -> (array)
    Takes a minimum and maximum observing frequency and returns the appropriate corresponding 
    coarse channels for CHORD'''
    min_chan = np.floor(freq_unit_strip(min_freq))
    max_chan = np.ceil(freq_unit_strip(max_freq))

    return np.arange(min_chan, max_chan + 1)

def read_catalogue(file):
    '''Function to open the galaxy catalogue and retrieve velocity and flux readings'''
    # Read Catalog:
    Catalog = np.loadtxt(file)

    # Galaxy parameters:
    MHI = Catalog[0]      # HI Mass - SolMass
    VHI = Catalog[1]      # HI Velocity - km/s
    i = Catalog[2]        # inclination - radians
    D = Catalog[3]        # Distance - Mpc
    W50 = Catalog[4]      # FWHM width - km/s
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
    Mfound = []; V = []; S = []; W = []; Wroots = []
    for j in range(sample_size):
        try_M, v, s, w, w_, _, _, _, _ = g.Generate_Spectra(MHI[j], VHI[j], i[j], D[j], a=a[j], b1=b1[j], b2=b2[j], c=c[j])
        Mfound.append(try_M)
        V.append(v)
        S.append(s)
        W.append(w)
        Wroots.append(w_)

    return V, S, z, ra, dec

def get_resampled_profiles(V, S, z, fine_freqs):
    '''Takes opened galaxy catalogue and returns finely re-sampled profiles in frequency space.
    Inputs:
        V, S (np.ndarray): velocity and flux obtained from read_catalogue function.
        nfreq (int): number of frequency points to be returned after re-sampling.
        midfreq (int): frequency at which to center the galaxy profiles
    Outputs: 
        freqs (np.ndarray): array of frequencies at which all profiles are sampled
        profiles (np.ndarray): the galaxy profiles from the catalogue '''
    
    # instantiating array to hold profiles
    resampled_profiles = np.zeros((len(S), len(fine_freqs)))

    # converting the units
    profile = GalaxyCatalog(V, S, z)

    for i in range(len(V)):

        new_prof = np.interp(fine_freqs, profile.obs_freq[i][::-1], profile.T[i][::-1]) 
        resampled_profiles[i] = new_prof

    # outputs them from high to low freq
    return resampled_profiles

def get_response_matrix(freqs, U, min_obs_freq = 1398, max_obs_freq = 1402, M = 4, N = 4096, viewmatrix = False):
    '''Gets the response matrix and the channels being observed on after upchannelization
    
    Inputs:
        freqs (np.ndarray): frequencies outputted by get_resampled_profiles function
        U (int): upchannelization factor, # of fine channels per coarse channel
        min_obs_freq, max_obs_freq (int): sets the observing range, determines coarse channels used
        M (int): # of taps for PFB
        N (int): chunk length for PFB
        viewmatrix (boolean): sets whether response matrix should be viewed
        
    Outputs:
        R (np.ndarray): response matrix, to be multiplied against profile for upchannelizing
        chans (np.ndarray): fine channel locations
        norm (np.ndarray): channelization envelope to be divided out for normalization '''

    # setting the coarse channels
    coarse_chans = get_chans(min_obs_freq, max_obs_freq)

    # stripping units and reshaping frequencies and channels
    f = np.reshape(freq_unit_strip(freqs[::-1]), (freqs.size, 1))
    c = np.reshape(coarse_chans, (1, len(coarse_chans))).astype(int)

    # generating response matrix - will eventually replace this step to simply load up the needed matrix file
    R = response_mtx(c, f, M, N, U)

    # visualizing the matrix:
    if viewmatrix == True:
        plt.figure(figsize = (10, 10), dpi = 200)
        plt.imshow(np.abs(R.real)**2, cmap = 'viridis', aspect='auto')
        plt.xlabel('Columns (f)')
        plt.ylabel('Rows (c)')
        plt.colorbar()
        plt.show()

    chans = np.arange(c.min()-0.5 + 1/(2*U), c.max() + 0.5, 1/U) 
    chans = freq_unit_add(chans)

    # removing frequency ripples from coarse channelization
    df = freqs[1] - freqs[0]
    dc = chans[1] - chans[0]
    freqs_null = np.arange(chans.min() - 2 * dc, chans.max() + 2 * dc, np.abs(df))
    f_null = np.reshape(freq_unit_strip(freqs_null), (freqs_null.size, 1))
    null = freqs_null * 0 + 1

    # generating response matrix for this new null function
    R_null = response_mtx(c, f_null, M, N, U)
    norm_unscaled = np.matmul(np.abs(R_null)**2, null)

    # generating scaling factors
    if U == 1: k = 1.216103148777748e-10
    elif U == 2: k = 7.841991167761238e-11
    elif U == 4: k = 3.195692185478832e-11
    elif U == 8: k = 1.5098060514380606e-11
    elif U == 16: k = 7.437551472089143e-12
    elif U == 32: k = 3.701749876806638e-12
    elif U == 64: k = 1.847847543734494e-12

    return R, chans, norm_unscaled * k

def upchannelize(profiles, U, R_filepath, norm_filepath):
    ''' Upchannelizes input profiles to get response on every channel
    Inputs:
        profiles (np.ndarray): profiles to be channelized, generated by get_resampled_profiles function
        U (int): upchannelization factor, # of fine channels per coarse channel
        R_filepath, norm_filepath (str): filepaths for outputs from get_response_matrix
        
    Outputs: 
        heights (np.ndarray): channelized profiles, index corresponds to profile # '''
    heights = []

    # loading in R, norm, and chans
    R = np.load(R_filepath)
    norm = np.load(norm_filepath)

    # getting response for each profile
    for i in range(len(profiles)):
        response = np.matmul(np.abs(R)**2, profiles[i][::-1])

        if U == 1: k = 1.216103148777748e-10
        elif U == 2: k = 7.841991167761238e-11
        elif U == 4: k = 3.195692185478832e-11
        elif U == 8: k = 1.5098060514380606e-11
        elif U == 16: k = 7.437551472089143e-12
        elif U == 32: k = 3.701749876806638e-12
        elif U == 64: k = 1.847847543734494e-12

        # removing ripples and scaling to correct height
        heights.append(np.array(response * k / norm))

    return heights

def channelize_catalogue(U, fstate, nside, catalogue_filepath, R_filepath, norm_filepath, fine_freqs, save_title):
    # getting velocity and flux from catalogue
    V, S, z, ra, dec = read_catalogue(catalogue_filepath)

    # resampling and converting into profiles in frequency space
    profiles = get_resampled_profiles(V, S, z, fine_freqs)

    # generating heights
    heights = upchannelize(profiles, U, R_filepath, norm_filepath)

    pol = 'full'
    map_catalog(fstate, np.flip(heights, axis=1), nside, pol, ra, dec, filename=save_title, write=True)

    return heights

def channelize_map(U, fstate, map_filepath, R_filepath, norm_filepath, fine_freqs, save_title):
    ''' Opening map '''
    f = h5py.File(map_filepath)
    Map = np.array(f['map'])  # the healpix map
    idx = f['index_map']
    ff = np.array(idx['freq'])
    freqs = np.array([ii[0] for ii in ff])  # the frequencies of each slice
    f.close()

    ''' re-sampling each pixel '''
    pixels = []
    npix = Map.shape[2]
    for i in range(npix):
        func = interpolate.interp1d(freqs, Map[:, 0, i], fill_value='extrapolate')
        pixels.append(func(fine_freqs))

    heights = upchannelize(pixels, U, R_filepath, norm_filepath)

    nfreq = fstate.frequencies.size
    npol = 4
    map_ = np.zeros((nfreq, npol, npix), dtype=np.float64)

    for i in range(len(heights)):
        map_[:, 0, i] = np.flip(heights[i])

    write_map(save_title, map_, fstate.frequencies, fstate.freq_width, include_pol=True)


