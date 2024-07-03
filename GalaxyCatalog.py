import numpy as np
from astropy import constants, units


class GalaxyCatalog(object):

    '''Class for galaxy catalogs
       Converts units from velocity to frequency and spectral flux density to temperature.'''

    def __init__(self, velocities, fluxes, redshifts, f_rest=1420.0, b_max=77):
        '''
        - velocities: <array>
          velocities (km/s) of the profiles for all galaxies in the catalog
        - fluxes: <array>
          spectral flux density (mJy) at each velocity for all galaxies in the catalog
        - redshifts: <array>
          cosmological redshifts for all galaxies in the catalog
        - f_rest: <float>
          rest frequency of the observation in MHz
          default set to 1420 MHz for 21 cm observation
        - b_max: <float>
          the maximum baseline of the interferometer in m
          for a single-dish telescope, b_max is the dish diameter
          default is for the CHORD 66-dish pathfinder
        '''
        self.v = velocities                              
        self.s = fluxes                                  
        self.z = redshifts 
        self.rest_freq = f_rest
        # frequency at center of profiles (MHz)                    
        self.mid_freq = self.rest_freq / (1 + self.z)    
        self.b_max = b_max

        self.convert_f()
        self.convert_T()

    def convert_f(self):
        '''Convert velocities in km/s to observed frequencies in MHz'''
        c = (constants.c.to(units.km/units.s)).value
        freqs = np.ones_like(self.v)

        for i, vel in enumerate(self.v):
          obs_freq_ = (self.rest_freq * (1-vel/c))
          shift_freq = self.mid_freq[i] - np.mean(obs_freq_)
          obs_freq = obs_freq_ + shift_freq
          freqs[i] = obs_freq

        self.obs_freq = freqs

    def convert_T(self):
        '''Convert spectral flux densitities in mJy to temperatures in K'''
        c = (constants.c.to(units.km/units.s))
        f = (self.obs_freq*units.MHz).to(1/units.s)
        self.wavelength = (c/f).to(units.m)
        ang_res = (self.wavelength) / (self.b_max*units.m)
        T = (self.s*units.mJy * self.wavelength**2 / (2*constants.k_B*ang_res**2)).to(units.K) 

        self.T = T