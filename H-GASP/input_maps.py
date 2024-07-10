import numpy as np
from savetools import make_map, map_catalog, write_map
from frequencies import FreqState
import Generate_HI_Spectra as g
from GalaxyCatalog import GalaxyCatalog
import healpy as hp
import h5py
import os


class HIGalaxies():
    '''Input maps from the galaxy catalog'''

    def __init__(self, catalog_path, f_start, f_end, nfreq):
        '''
        - catalog_path: <str>
          the path and name of the catalog to use
        - f_start, f_end: <float>
          largest and smallest frequencies of your beam transfer matrices
        - nfreq: <int>
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
          path and filename to which we save the input maps.
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
        - nfreq: <int>
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
    '''Input healpix maps for other sky components given in cora package'''

    def __init__(self, f_start, f_end, nfreq, nside, output_directory):
        '''
        - f_start, f_end: <float>
          largest and smallest frequencies of your beam transfer matrices
        - nfreq: <int>
          number of frequencies
        - nside: <int>
          defining the resolution of the created healpix map. nside = 2^n.
        - output_directory: <str>
          path to which we save the input foreground maps
        '''
        self.f_start = f_start
        self.f_end = f_end
        self.nfreq = nfreq
        self.nside = nside
        self.output_directory = output_directory
        self.pol = 'full'

    def get_component(self, component_name):
        '''runs cora-makesky command to simulate input maps for "component_name"'''

        command = '{} --nside={} --freq {} {} {} --pol={} --filename={}'.format(component_name,
                                                                                  self.nside,
                                                                                  self.f_start,
                                                                                  self.f_end,
                                                                                  self.nfreq,
                                                                                  self.pol,
                                                                                  self.output_directory+component_name+'.h5')
        os.system('cora-makesky ' + command)

    def get_maps(self, component_names_list):
        '''
        Gets all simulated maps for sky components and combines them for a full foreground map.

        Inputs
        ------
        - component_names_list: <list of strings>
          a list containing the sky components to be simulated with cora
          options are: 21cm, pointsource, galaxy, foreground
          for more info on each, read the cora foregound documentation

        Outputs
        -------
        - creates an individual healpix map for each of the desired components
        - saves an additional file in output_directory with all foreground components combined
          called foregrounds_all.h5
        '''
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

