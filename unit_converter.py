import numpy as np
from astropy import constants, units


class GalaxyProfile(object):

    '''Contains a single galaxy profile object
       Has a frequency equivalent axis and an intensity equivalent axis.
       Converts units from velocity to frequency and spectral flux density to temperature'''

    def __init__(self, data, mid_freq=1400, redshift=None):

        '''
        Inputs
        ------
        data <ndarray>:
            contains the galaxy profile data
            shape is (n_fluxes, 2)
        mid_freq <float>:
            frequency to center the profile on
            units of MHz
        '''

        ## TO DO: need to add a section to check that the units 
        ## and shapes are correct or at least exist
        self.data = data
        self.velocity = data[:,0]          # array of velocities in profile (km/s)
        self.flux = data[:,1]              # array of fluxes at each velocity (mJy)
        self.rest_freq = 1420              # rest frequency of emission (MHz)

        if redshift:
            self.z = redshift
            self.mid_freq = self.rest_freq / (1 + self.z)      # frequency at center of profile (MHz)

        else:
            self.mid_freq = mid_freq           # frequency at center of profile (MHz)
        
        
    def convert_vel_freq(self):

        '''Converts velocity axis (km/s) into frequency axis (MHz) centred at mid_freq'''
        
        from astropy import constants, units

        c = (constants.c.to(units.km/units.s)).value
        obs_freq = (self.rest_freq * (1 - self.velocity/c))
        shift_freq = self.mid_freq - np.mean(obs_freq)
        self.obs_freq = obs_freq + shift_freq

    def convert_flux_temp(self):

        '''Converts spectral flux density axis (mJy) into temperature axis (K)'''

        from astropy import constants, units

        self.convert_vel_freq()
        
        c = constants.c
        f = (self.obs_freq*units.MHz).to(1/units.s)
        self.wavelength = (c/f).to(units.m)
        self.resolution = (self.wavelength) / (87.8872*units.m)   # approximate resolution of 6m dish -- update for more accurate beam?
        
        self.T = (self.flux*units.mJy * self.wavelength**2 / (2*constants.k_B*self.resolution**2)).to(units.K)

    def get_sampling(self):
        
        '''Gets velocity sampling and spectral resolution'''

        self.convert_vel_freq()

        self.dv = np.abs(self.velocity[1] - self.velocity[0])       # sampling in velocity space - noticed varied along the x-axis - why is that?
        self.dfreq = np.abs(self.obs_freq[1] - self.obs_freq[0])    # frequency resolution in MHz

        
    def convert_units(self):

        '''Converts both axis of interest into cosmological units'''
        
        self.convert_vel_freq()
        self.convert_flux_temp()
        self.get_sampling()


    def plot(self):
        
        '''plots the galaxy profile'''
        
        import matplotlib.pyplot as plt
        
        self.convert_units()
        
        plt.figure(figsize=(10,7))
        plt.plot(self.obs_freq, self.T, color='black')
        plt.xlabel('MHz', fontsize=20)
        plt.ylabel('K', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Galaxy Profile', fontsize=20)
        plt.show()


class GalaxyCatalog(object):

    '''Same as GalaxyProfile but for an entire catalog of HI galaxies.'''

    def __init__(self, velocities, fluxes, redshifts):

        self.v = velocities                              # array of velocities in profile (km/s) for all HI galaxies
        self.s = fluxes                                  # array of fluxes at each velocity (mJy) for all HI galaxies
        self.z = redshifts                               # array of redshifts of all HI galaxies in catalog
        self.rest_freq = 1420                            # rest frequency of emission (MHz)
        self.mid_freq = self.rest_freq / (1 + self.z)    # frequency at center of profile (MHz)
        
        # setting up converted units
        self.convert_vel_freq()
        self.convert_flux_temp()
        self.get_sampling()


    def convert_vel_freq(self):

        '''Converts velocity axis (km/s) into frequency axis (MHz) centred at mid_freq'''
        
        c = (constants.c.to(units.km/units.s)).value

        freqs = np.ones_like(self.v)
        
        for i, vel in enumerate(self.v):
            obs_freq_ = (self.rest_freq * (1 - vel/c))
            shift_freq = self.mid_freq[i] - np.mean(obs_freq_)
            obs_freq = obs_freq_ + shift_freq
            freqs[i] = obs_freq

        self.obs_freq = freqs

    def convert_flux_temp(self):

        '''Converts spectral flux density axis (mJy) into temperature axis (K)'''


        c = constants.c
        f = (self.obs_freq*units.MHz).to(1/units.s)
        self.wavelength = (c/f).to(units.m)
        self.resolution = (self.wavelength) / (63.5*units.m)   # approximate resolution for 6x6 array
        
        self.T = (self.s*units.mJy * self.wavelength**2 / (2*constants.k_B*self.resolution**2)).to(units.K)

    def convert_mJy_to_K_sr(self):
        # not using this one, I really don't think it does what I think it does
        # I'm not integrating over pixels to find the flux, I'm assuming it goes within the CHORD beam completely
        """
        Calculate multiplicative factors to convert [mJy] to [K sr].

        Parameters
        ----------
        freqs: 'astropy.Quantity' or array_like of float
            Frequencies
        s: 'astropy.Quantity' or array_like of float with same dimentions of freqs
            spectral flux density, assumed to be in mJy if not a Quantity.

        Returns
        -------
        conv_factor: 'astropy.Quantity'
            Conversion factor(s) to go from [Jy] to [K sr]. Shape equal to shape of freqs.
        """
        freqs = self.obs_freq*units.MHz
        freqs = np.atleast_1d(freqs)

        s = self.s*units.mJy

        equiv = units.brightness_temperature(freqs, beam_area=1*units.sr)
        K_sr = (s).to(units.K, equivalencies=equiv) * units.sr

        self.T_sr = K_sr

    def get_sampling(self):
        
        '''Gets velocity sampling and spectral resolution'''

        self.convert_vel_freq()

        self.dv = np.abs(self.v[1] - self.v[0])       # sampling in velocity space - noticed varied along the x-axis - why is that?
        self.dfreq = np.abs(self.obs_freq[1] - self.obs_freq[0])    # frequency resolution in MHz

        
    #def convert_units(self):

        #'''Converts both axis of interest into K/MHz units'''
        
        #self.convert_vel_freq()
        #self.convert_flux_temp()
        #self.get_sampling()
        