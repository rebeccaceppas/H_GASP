import numpy as np
from savetools import make_map, map_catalog, write_map
from frequencies import FreqState
import channelization_functions as cf
import Generate_HI_Spectra as g
from GalaxyCatalog import GalaxyCatalog
import matplotlib.pyplot as plt
import subprocess
import healpy as hp
import h5py


class HIGalaxies():
    '''Input maps from the galaxy catalog'''

    def __init__(self, catalog_path, f_start, f_end, nfreq):
        '''
        - catalog_path: <str>
          the path and name of the catalog to use
        - f_start, f_end: <float>
          largest and smallest frequencies of your beam transfer matrices
        - nfreq: <inf>
          number of frequencies
        '''
        self.catalog = catalog_path
        self.f_start = f_start
        self.f_end = f_end
        self.nfreq = nfreq

        self.pol = "full"

    def get_map(self, nside, output_filepath, ngals=None, ras=[], decs=[], T_brightness=[], max_baseline=77, seed=0):
        '''
        Creates an input map with the desired HI galaxies from the catalog

        Inputs
        ------
        - nside: <int>
          defining the resolution of the created healpix map. nside = 2^n.
        - output_filepath: <str>
          path and filename to which we save the input synthesized beam map.
        - ngals: <int>
          number of galaxies to inject into the map.
          default is None which injects all galaxies from the catalog.
        - ras, decs: <list of floats>
          right ascension and declination of each galaxy in degrees
          default is empty list which injects them at a location specified in the catalog
        - T_brightness: <list of floats>
          peak brightness of your sources.
          default is mepty list which respects peak brightness given by the catalog.
        - max_baseline: <float>
          the longest baseline of the instrument in m
          used for the conversion between Jy and K
          default is 77m which is approximately for the CHORD pathfinder
        - seed: <int>
          seed for the random selection of ngals galaxies
        '''
        # checking the arrays satisfy basic conditions
        if len(ras) > 0:
            assert len(decs)==len(ras), """Your positions ras and decs must have the same length but have 
                                           {} and {} respectively.""".format(len(ras), len(decs))

            assert len(ras)==ngals, """Your positions should match the number of galaxies you want to inject
                                       but are {} and {} respectively.""".format(len(ras), ngals)
            
        if len(T_brightness) > 0:
            assert ngals==len(T_brightness), """Your temperature brightness T_brightness must have the same length as your number of galaxies 
            but they have {} and {} respectively.""".format(len(T_brightness), ngals)

        fstate = FreqState()
        fstate.freq = (self.f_start, self.f_end, self.nfreq)

        # getting spectra
        V, S, z, ra, dec = get_spectra(self.catalog, ngals=ngals, seed=seed)

        # converting to K vs MHz and resampling to desired frequencies
        resampled_profiles = np.zeros((len(S), len(fstate.frequencies)))

        profile = GalaxyCatalog(V, S, z, b_max=max_baseline)
        freqs = profile.obs_freq
        temps = profile.T

        for i in range(len(V)):
            resamp = np.interp(fstate.frequencies, freqs[i][::-1], temps[i][::-1])
            
            # very computational artefacts only, set to 0
            if np.all(resamp.value < 1e-15) :
                resampled_profiles[i] = np.zeros_like(resamp)
            else:
                resampled_profiles[i] = resamp

        if len(ras) > 0:
            ra, dec = ras, decs

        if len(T_brightness) > 0:

            resampled_profiles = resampled_profiles * np.array(T_brightness).reshape((ngals, -1)) / np.max(resampled_profiles, axis=1).reshape((ngals, -1))

        map_catalog(fstate,
                    resampled_profiles,
                    nside,
                    self.pol,
                    ra, dec,
                    filename=output_filepath,
                    write=True)

class SynthesizedBeam():
    '''Healpix map to simulate a synthesized beam through map-maker'''

    def __init__(self, f_start, f_end, nfreq=2):
        '''
        - f_start, f_end: <float>
          largest and smallest frequencies of your beam transfer matrices
        - nfreq: <inf>
          number of frequencies in your beam transfer matrices
          minimum is 2 if you don't care about frequency evolution
          otherwise, match to the number of frequencies of your previous specifications
        '''
        self.f_start = f_start
        self.f_end = f_end
        self.nfreq = nfreq
        self.pol = 'full'

    def get_map(self, nside, ra, dec, output_directory, output_filename=None, T_brightness=1.0):
        '''
        Creates an input map with a single source for simulating synthesized beams with the dirty map maker

        Inputs
        ------
        - nside: <int>
          defining the resolution of the created healpix map. nside = 2^n.
        - ra, dec: <float>
          right ascension and declination in degrees
        - output_filepath: <str>
          path and filename to which we save the input synthesized beam map
          default None will name it based on ra and dec.
        - T_brightness: <float>
          brightness of your source. Default is 1.
        '''
        fstate = FreqState()
        fstate.freq = (self.f_start, self.f_end, self.nfreq)

        temps = np.ones(self.nfreq)*T_brightness

        if output_filename is None:
            output_filename = 'input_RA{0:.2f}_DEC{1:.2f}.h5'.format(ra, dec)

        input_map = make_map(fstate.frequencies,
                             fstate.freq_width,
                         temps,
                         nside, self.pol,
                         ra, dec,
                         write=True,
                         filename=output_directory + output_filename,
                         new=True, existing_map=None)

class Foregrounds():
    '''submitting jobs for the cora makesky components
    
    want to save one for each one they ask for individually 
    and one complete one with all the ones combined?
    '''

    def __init__(self, f_start, f_end, nfreq, nside, output_directory):
        
        self.f_start = f_start
        self.f_end = f_end
        self.nfreq = nfreq
        self.nside = nside
        self.output_directory = output_directory
        self.pol = 'full'

    def get_component(self, component_name):

        command = '{} --nside={} --freq {} {} {} --pol={} --filename={}'.format(component_name,
                                                                                  self.nside,
                                                                                  self.f_start,
                                                                                  self.f_end,
                                                                                  self.pol,
                                                                                  self.output_directory+component_name+'.h5')

        subprocess.call(['cora-makesky', command])

    def get_maps(self, component_names_list):
        '''gets all simulated maps and combines them for a full foreground map'''

        fstate = FreqState()
        fstate.freq = (self.f_start, self.f_end, self.nfreq)

        foregrounds_all = np.zeros((self.nfreq, 4, hp.nside2npix(self.nside)))

        for component in component_names_list:
            
            filename = self.output_directory + component + '.h5'
            self.get_component(component)
            foregrounds_all += open_map(filename)

        write_map(self.output_directory+'foregrounds_all.h5',
                  foregrounds_all,
                  fstate.frequencies,
                  fstate.freq_width)
        





    


def get_spectra(filepath, ngals=None, seed=0):
    '''
    Function to open the galaxy catalogue, retrieve velocity and flux readings.

    Inputs
    ------
    - filepath: <str>
      path to the text file containing the catalog
    - ngals: <int> or <None>
      number of galaxies to include from the catalog
      default is None which includes all galaxies from catalog
    - seed: <int>
      seed for the random sample of galaxies to select


    Outputs
    -------
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

    V = []; S = []; ras = []; decs = []; zs = []

    if ngals is None:
        ngals = len(MHI)
        print('Generating spectra for {} galaxies.'.format(ngals))
        
        for j in range(ngals):
            _, v, s, _, _, _, _, _, _ = g.Generate_Spectra(MHI[j], VHI[j], i[j], D[j], 
                                                            a=a[j], b1=b1[j], b2=b2[j], c=c[j])
            V.append(v)
            S.append(s)
            ras.append(ra[j])
            decs.append(dec[j])
            zs.append(z[j])

    else:
        np.random.seed(seed)
        js = np.random.choice(np.arange(0, len(MHI)), size=ngals, replace=False)
        print('Generating spectra for {} galaxies.'.format(ngals))

        for j in js:
            _, v, s, _, _, _, _, _, _ = g.Generate_Spectra(MHI[j], VHI[j], i[j], D[j], 
                                                            a=a[j], b1=b1[j], b2=b2[j], c=c[j])
            V.append(v)
            S.append(s)
            ras.append(ra[j])
            decs.append(dec[j])
            zs.append(z[j])

    return V, S, np.array(zs), ras, decs


def open_map(filepath):

    f = h5py.File(filepath)
    m = f['map'][:]
    f.close()

    return m




##### old functions ######



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



