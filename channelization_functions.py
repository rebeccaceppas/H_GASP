# Code developed with Olivia Pereira #

import numpy as np
import numexpr as ne 
import matplotlib.pyplot as plt
from GalaxyCatalog import GalaxyCatalog 
import Generate_HI_Spectra as g
import h5py
from savetools import write_map, map_catalog
from scipy import interpolate

############################# helper functions ##############################

def freq_unit_strip(f, fmin=300, number_channels=2048, sampling_rate=0.417):
    '''
    Strips quantities in frequency-space to become unitless
    
    Inputs
    - f: <array>
      array of frequencies in MHz
    - fmin: <float>
      minimum instrumental frequency. default is CHORD's 300 MHz
    - number_channels: <int>
      number of channels. default is CHORD's 2048
    - sampling_rate: <float>
      time stream sampling rate in ns. default is CHORD's 0.417 ns

    Outputs
    - f_bar: <array>
      unitless frequency array
    '''
    N = 2*number_channels
    f_bar = (f - fmin) * N * sampling_rate * 0.001
    return f_bar

def freq_unit_add(f_bar, fmin=300, number_channels=2048, sampling_rate=0.417):
    '''
    Adds frequency units (MHz) to unitless quantities
    
    Inputs
    - f_bar: <array>
      array of unitless frequencies
    - fmin: <float>
      minimum instrumental frequency. default is CHORD's 300 MHz
    - number_channels: <int>
      number of channels. default is CHORD's 2048
    - sampling_rate: <float>
      time stream sampling rate in ns. default is CHORD's 0.417 ns

    Outputs
    - f: <array>
      frequency array with units of MHz
    '''
    N = 2*number_channels
    f = f_bar / (N * sampling_rate * 0.001) + fmin
    return f

def get_chans(fmax_chan, fmin_chan, fmin=300, number_channels=2048, sampling_rate=0.417):
    '''
    Gets the coarse channel indices for a range of observed frequencies given
    some telescope sampling and spectral properties.
    
    Inputs
    - fmax_chan: <float>
      maximum frequency to consider for channels in MHz
    - fmin_chan: <float>
      minimum observed frequency to consider for channels in MHz
    - fmin: <float>
      minimum instrumental frequency. default is CHORD's 300 MHz
    - number_channels: <int>
      number of channels. default is CHORD's 2048
    - sampling_rate: <float>
      time stream sampling rate in ns. default is CHORD's 0.417 ns

    Outputs
    - array corresponding to the coarse channel indices
      for the chosen frequency range and instrumental parameters
    '''
    min_chan = np.floor(freq_unit_strip(fmin_chan, fmin, number_channels, sampling_rate))
    max_chan = np.ceil(freq_unit_strip(fmax_chan, fmin, number_channels, sampling_rate))
    return np.arange(min_chan, max_chan + 1)

def get_fine_freqs(coarse_frequencies):
    '''
    Helper function to get fine frequencies 
    for resampling spectra before up-channelization.
    It adds some padding and makes the resolution 3 times finer.

    Inputs
    - coarse_frequencies: <array>
      frequencies in MHz in descending order (max to min)

    Outputs
    - finer frequencies within that range for up-channelization
    '''
    fmax = np.max(coarse_frequencies)+2
    fmin = np.min(coarse_frequencies)-2
    dc = coarse_frequencies[1] - coarse_frequencies[0] 
    return np.arange(fmax, fmin, dc / 3) 

def window(index, taps=4, N=4096):
    '''
    Sinc-Hanning window function

    Inputs
    ------
    - index: <array of int>
    - taps: <int>
      number of taps
    - N: <int>
      2 x number of channels. default is CHORD's 4096

    Outputs
    -------
    - W: <array>
      Sinc-Hanning windown function 
    '''
    W = (np.cos(np.pi * (index-taps*N/2)/(taps*N-1)))**2 * np.sinc((index-taps*N/2)/N)
    return W

def exponential_chan(s, mtx, number_channels=2048): 
    '''
    Calculates exponential term of first-round PFB

    Inputs
    - s: <array>
      indices for summation
    - mtx: <array>
      matrix containing (c - f)
    - number_channels: <int>
      number of channels. default is CHORD's 2048

    Outputs
    - exponential e^(-2*i*pi*mtx*s/N)
    '''
    # reshaping (coarse chans, nfreq) -> (coarse chans, nfreq, 1)
    mtx = np.reshape(mtx, (mtx.shape[0], mtx.shape[1], 1))
    N = 2*number_channels
    # shapes = (coarse chans, nfreqs, 1) x (1, taps*N) 
    #        = (coarse chans, nfreqs, taps*N)
    v = np.matmul(mtx, s)
    exponent = -2j * np.pi * v / N
    return ne.evaluate("exp(exponent)")

def weight_chan(cf, taps=4, number_channels=2048): 
    '''
    First-round PFB channelization
    
    Inputs
    ------
    - cf: <array>
      entries are c - f for relevant channels c and frequencies f
    - taps: <int>
      number of taps
    - number_channels: <int>
      number of channels. default is CHORD's 2048

    Outputs
    -------
    array of shape (2*number_channels, nfreq) containing the first-round PFB output
    ''' 
    N = 2*number_channels
    # array containing the indices over which we need to sum the DFT over 
    # shape = (1, M*N)
    j = np.reshape(np.arange(taps*N), (1, taps*N)) 

    # shape = (coarse chans, nfreq, M*N)
    summation = window(j, taps, N) * exponential_chan(j, cf, number_channels)

    # collapsing to shape (coarse chans, nfreq)
    return np.sum(summation, axis = 2)

def exponential_upchan(s, mtx):
    '''
    Calculates exponential term of second-round PFB

    Inputs
    - s: <array>
      indices for summation
    - mtx: <array>
      matrix containing (c*u - f) for each coarse channel c

    Outputs
    - exponential e^(i*pi*mtx*k)
    '''
    # shape = (U, nfreq, 1)
    mtx = np.reshape(mtx, (mtx.shape[0], mtx.shape[1], 1))

    # shapes = (U, nfreqs, 1) x (1, taps*U) = (coarse chans, nfreqs, taps*U)
    v = np.matmul(mtx, s)
    exponent = np.pi * 1j * v
    return ne.evaluate("exp(exponent)")

def weight_upchan(cfu, U, taps=4):
    '''
    Second-round PFB channelization
    
    Inputs
    ------
    - cfu: <array>
      entries are c - f for relevant fine channels u and frequencies f in coarse channels c
      has ((U-1)/U - 2u/U + 2f) entries
    - U: <int>
      up-channelization factor, U = 2^n.
    - taps: <int>
      number of taps

    Outputs
    -------
    array of shape (U, nfreq) containing the second-round PFB output
    ''' 
    # creating an array containing the indices over which we need to sum the DFT over, shape = (1, M*U)
    k = np.reshape(np.arange(taps*U), (1, taps*U))

    # getting window function and exponential bit of the DFT which need to be multiplied together
    # result has shape (coarse chans, nfreq, M*U)
    summation = window(k, taps, U) * exponential_upchan(k, cfu)

    # summation collapses shape back down to shape = (U, nfreq)
    return np.sum(summation, axis = 2) 

def scaling(U):
    '''pre-computed overal scaling factors
       for each up-channelization factor U'''
    if U == 1: k = 1.216103148777748e-10
    elif U == 2: k = 7.841991167761238e-11
    elif U == 4: k = 3.195692185478832e-11
    elif U == 8: k = 1.5098060514380606e-11
    elif U == 16: k = 7.437551472089143e-12
    elif U == 32: k = 3.701749876806638e-12
    elif U == 64: k = 1.847847543734494e-12
    return k

############################# main computations ##############################

def response_mtx(c, f, U, taps=4, number_channels=2048):
    '''
    Calculates response matrix which can be multiplied against input frequencies 
    to give response on channels
    
    Inputs:
    - c: <array>
      indices of the coarse channels to channelize
      require shape (1, # of coarse channels)
    - f: <array>
      large number of frequencies to simulate continuous 'real' spectrum
      can be calculated using function get_fine_freqs
      require shape (1, # of frequencies)
    - U: <int>
      up-channelization factor, U = 2^n 
    - taps: <int>
      number of taps
    - number_channels: <int>
      number of channels. default is CHORD's 2048  
 

    Outputs
    - response matrix of shape (number of fine channels x number of frequencies / length of profile)
      can be multiplied against a galaxy profile/spectrum to yield the response on 
      each fine channel to the profile as a whole
    '''
    # creating coarse channelization matrix, size = (number of coarse chans x nfreq)
    # where each entry is (c-f) of the relevant channel and frequency
    submtx_chan = np.tile(c, [f.shape[0], 1]).T - f[:,0]

    # passing each element in the matrix through the coarse channelization algorithm
    submtx_chan = weight_chan(submtx_chan, taps, number_channels)

    # reshaping the resulting matrix so that we get U identical rows per coarse channel
    mtx_chan = np.repeat(submtx_chan, U, axis = 0)  

    # creating fine upchannelization matrix, size = (U x nfreq)
    submtx_upchan = np.tile(np.arange(U), [f.shape[0], 1]).T

    # making it so that every entry corresponds to the expression needed in the exponential term 
    # of the upchannelization weight function
    submtx_upchan = (U-1) / U - 2*submtx_upchan / U + 2*f[:,0]

    # passing through upchannelization algorithm
    submtx_upchan = weight_upchan(submtx_upchan, U, taps)  

    # reshaping so that we get repeating blocks from u = 1...U for each coarse channel
    mtx_upchan = np.tile(submtx_upchan, (len(c[0]), 1)) # tiling resulting matrix to correct shape

    # returning the combined response matrix where coarse and fine channelization weights are multiplied
    return np.multiply(mtx_chan, mtx_upchan)

def get_response_matrix(freqs, U, min_freq , max_freq, taps=4, number_channels=2048, viewmtx=False):
    '''
    Generates the response matrix and the channels being observed on after up-channelization
    
    Inputs
    - freqs: <array>
      large number of frequencies to simulate continuous 'real' spectrum
      can be calculated using function get_fine_freqs
    - U: <int>   
      up-channelization factor, U = 2^n
    - min_freq, max_freq: <float>
      minimum and maximum frequencies to observe
      sets the observing range, determines coarse channels used
    - M: <int>
      number of taps. defalt is 4
    - N: <int>
      2*number of channels. default is CHORD's 2*2048 channels
    - viewmatrix: <bool>
      if True produces plot of the response matrix. default is False
        
    Outputs
    - R: <array>
      response matrix, to be multiplied against profile for upchannelizing
    - frequencies: <array> 
      frequencies for each channel with response after up-channelization
    - norm: <array> 
      channelization envelope to be divided out of up-channelized profiles 
      to remove modulation
      '''
    # setting the coarse channels and retrieving the scaling factor
    k = scaling(U)
    coarse_chans = get_chans(max_freq, min_freq)

    # stripping units and reshaping frequencies and channels
    f = np.reshape(freq_unit_strip(freqs[::-1]), (freqs.size, 1))
    c = np.reshape(coarse_chans, (1, len(coarse_chans))).astype(int)

    # generating response matrix
    R = response_mtx(c, f, U, taps, number_channels)
    chans = np.arange(c.min()-0.5 + 1/(2*U), c.max() + 0.5, 1/U) 
    frequencies = freq_unit_add(chans)

    # flat spectrum to get normalization vector
    flat = np.ones_like(freqs)
    norm_unscaled = np.matmul(np.abs(R)**2, flat[::-1])

    # visualizing the matrix:
    if viewmtx == True:
        plt.figure(figsize = (10, 6))
        plt.imshow(np.abs(R.real)**2, cmap = 'viridis', aspect='auto')
        plt.xlabel('Columns (f)')
        plt.ylabel('Rows (c)')
        plt.colorbar()
        plt.show()

    return R, frequencies, norm_unscaled * k

############################# general usage ##############################

def upchannelize(spectra, U, R_path, norm_path, freq_path):
    '''
    Up-channelizes input spectra to get response on every channel
    
    Inputs
    - spectra: <array>
      shape is (# of spectra, # of frequencies) or # of frequencies for single input
      spectra to be up-channelized
      must correspond to the same frequency range as the one used to compute 
      R and norm in get_response_matrix
    - U: <int>
      up-channelization factor, U = 2^n
    - R_path, norm_path, freq_path: <str>
      paths to response matrix R, normalization vector norm,
      and output frequencies from get_response_matrix
        
    Outputs
    - responses: <array>
      up-channelized spectra
      shape is (# spectra, U x # coarse channels)
    - frequencies: <array>
      frequencies for each channel with response after up-channelization
    '''
    responses = []

    # loading in R, norm, and scaling factor
    R = np.load(R_path)
    norm = np.load(norm_path)
    frequencies = np.load(freq_path)
    k = scaling(U)

    spectra = np.array(spectra)
    dimensions = spectra.ndim

    # if more than one spectrum, get response for each separately
    if dimensions > 1:
        for i in range(len(spectra)):
            r = np.matmul(np.abs(R)**2, spectra[i][::-1])
            responses.append(np.array(r * k / norm))
    else:
        r = np.matmul(np.abs(R)**2, spectra[::-1])
        responses = np.array(r * k / norm)
    return responses, frequencies

############################# application: up-channelizing healpix map ##############################

def channelize_map(U, map_path, R_path, norm_path, freq_path, fine_freqs, output_path):
    '''
    Up-channelizes an entire healpix map
    Works for the file formats of the existing CHORD pipeline

    Inputs
    - U: <int>
      up-channelization factor, U = 2^n
    - map_path: <str>
      path to the input map to up-channelize (.h5 file)
    - R_path, norm_path, freq_path: <str>
      paths to response matrix R, normalization vector norm,
      and output frequencies from get_response_matrix
    - fine_freqs: <array>
      large number of frequencies to simulate continuous 'real' spectrum
      calculated using function get_fine_freqs
    - output_path: <str>
      filename and path to save the new up-channelized map
      should be a .h5 file

    Outputs
    - produces and saves up-channelized map to output_path
    '''
    f = h5py.File(map_path)
    Map = np.array(f['map'])
    idx = f['index_map']
    ff = np.array(idx['freq'])
    freqs = np.array([ii[0] for ii in ff])
    f.close()

    pixels = []
    npix = Map.shape[2]
    for i in range(npix):
        func = interpolate.interp1d(freqs, Map[:, 0, i], fill_value='extrapolate')
        pixels.append(func(fine_freqs))

    responses, frequencies = upchannelize(pixels, U, R_path, norm_path, freq_path)

    nfreq = len(frequencies)
    fwidth = np.abs(frequencies[0] - frequencies[1])
    npol = 4
    map_ = np.zeros((nfreq, npol, npix), dtype=np.float64)

    for i in range(len(responses)):
        map_[:, 0, i] = np.flip(responses[i])

    # we flip the response and frequencies so the slices go from high to low frequency
    write_map(output_path, map_, np.flip(frequencies), fwidth, include_pol=True)

############################# application: up-channelizing catalogue of galaxy profiles ##############################

def read_catalogue(filepath):
    '''
    Function to open the galaxy catalogue, retrieve velocity and flux readings.

    Inputs
    - filepath: <str>
      path to the text file containing the catalog


    Outputs
    - V: <array>
      the velocities for each profile in km/s
    - S: <array>
      spectral flux density for each profile in mJy
    - ra: <array>
      Right Ascension of each source
    - dec: <array>
      Declination of each source
    '''
    Catalogue = np.loadtxt(filepath)
    MHI = Catalogue[0]      # HI Mass - SolMass
    VHI = Catalogue[1]      # HI Velocity - km/s
    i = Catalogue[2]        # inclination - radians
    D = Catalogue[3]        # Distance - Mpc
    W50 = Catalogue[4]      # FWHM width - km/s
    z = Catalogue[5]        # Redshift 
    ra = Catalogue[6]       # Right Ascension - Degrees
    dec = Catalogue[7]      # Declination - Degrees

    # Busy function parameters:
    a = Catalogue[8]        # Controls peak 
    b1 = Catalogue[9]       # Controls height of one peak in double-peak profile
    b2 = Catalogue[10]      # Controls height of other peak in double-peak profile
    c = Catalogue[11]       # Controls depth of trough

    sample_size = len(Catalogue[0])
    V = []; S = []
    for j in range(sample_size):
        _, v, s, _, _, _, _, _, _ = g.Generate_Spectra(MHI[j], 
                                                            VHI[j], 
                                                            i[j], 
                                                            D[j], 
                                                            a=a[j], b1=b1[j], b2=b2[j], c=c[j])
        V.append(v)
        S.append(s)

    return V, S, z, ra, dec

def get_resampled_profiles(V, S, z, fine_freqs, b_max=77):
    '''
    Re-samples profiles finely in frequency space
    
    Inputs
    - V, S: <array>
      velocity (km/s) and flux (mJy) obtained from read_catalogue function
    - z: <array>
      redshifts of all sources in the catalogue
    - fine_freqs: <array>
      large number of frequencies to simulate continuous 'real' spectrum
      calculated using function get_fine_freqs
    - b_max: <float>
      the maximum baseline of the interferometer in m
      for a single-dish telescope, b_max is the dish diameter
      default is for the CHORD 66-dish pathfinder b_max = 77 m
      
    Outputs
    - resampled_profiles: <array>
    all galaxy profiles from the catalogue resampled at the fine frequencies 
    in K per MHz channel)
    '''
    # instantiating array to hold profiles
    resampled_profiles = np.zeros((len(S), len(fine_freqs)))

    # converting km/s -> MHz given z and mJy -> MHz
    # outputs them from high to low freq
    profile = GalaxyCatalog(V, S, z, b_max=b_max)

    for i in range(len(V)):
        new_prof = np.interp(fine_freqs, profile.obs_freq[i][::-1], profile.T[i][::-1]) 
        resampled_profiles[i] = new_prof

    return resampled_profiles 

def channelize_catalogue(U, fstate, nside, catalogue_path, R_path, norm_path, freq_path, fine_freqs, output_path,  b_max=77):
    '''
    Up-channelized the profiles from a galaxy catalog

    Inputs
    - U: <int>
      up-channelization factor, U = 2^n
    - fstate: <FreqState object>
      object containing the frequency specifications for the
      up-channelized profiles
    - nside: <int>
      specifies the resolution of the healpix map produced, nside = 2^m
    - catalogue_path: <str>
      path to the desired galaxy catalog
    - R_path, norm_path, freq_path: <str>
      paths to response matrix R, normalization vector norm,
      and output frequencies from get_response_matrix
    - fine_freqs: <array>
      large number of frequencies to simulate continuous 'real' spectrum
      calculated using function get_fine_freqs
    - output_path: <str>
      filename and path to save the new up-channelized map
      should be a .h5 file
    - b_max: <float>
      the maximum baseline of the interferometer in m
      for a single-dish telescope, b_max is the dish diameter
      default is for the CHORD 66-dish pathfinder b_max = 77 m

    Outputs
    - produces and saves up-channelized map to output_path
    - responses: <array>
      up-channelized profiles
      shape is (# profiles in catalogue, U x # coarse channels)
    '''
    
    # getting velocity and flux from catalogue
    V, S, z, ra, dec = read_catalogue(catalogue_path)

    # resampling and converting into profiles in frequency space
    profiles = get_resampled_profiles(V, S, z, fine_freqs, b_max=b_max)

    # generating responses
    responses = upchannelize(profiles, U, R_path, norm_path, freq_path)
    pol = 'full'
    map_catalog(fstate, np.flip(responses, axis=1), nside, pol, ra, dec, filename=output_path, write=True)

    return np.flip(responses)