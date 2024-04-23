
## This script generates a random HI spectrum, for a given:
##        1. HI mass         (MHI in SolMass)
##        2. Velocity width  (VHI in km/s) 
##        3. Inclination     (i in radians)
##        4. Distance        (D in Mpc)

import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import numpy as np
from random import choices
from scipy import special
from numpy import diff
from scipy.interpolate import UnivariateSpline
from scipy.signal import fftconvolve

def Busy_general(x, a, b1, b2, xe, xp, c, w, n):
    ''' This is the functional definition for a General Busy Function:
        Reference: https://ui.adsabs.harvard.edu/abs/2014MNRAS.438.1176W/abstract (Section 4.1, Equation 4)'''
    
    err_p = special.erf(b1*(w+x-xe)) + 1
    err_m = special.erf(b2*(w-x+xe)) + 1
    pbola = (c*((x-xp)**n)) + 1
    
    return (a/4)*err_p*err_m*pbola

def integrate_profile(V, S):
    ''' Helper function which integrates an HI profile. The 'V' parameter is the velocity axis in km/s. The 'S' paramater is the flux in mJy'''

    S = (S*u.mJy).to(u.Jy)
    return np.trapz(S.value, dx=diff(V))  # trapz is a numeric integrator in python using trapezoid rule, diff(V) if dV

def get_MHI(V, S, D):
    ''' Helper function which converts the integrated HI profile to HI Mass using equation: M_HI = 2.356x10^5 * (D^2) * S_tot '''

    int_S = integrate_profile(V, S)
    return 2.356e5 * (D**2) * int_S

def find_FWHM(x, y):
    ''' Helper function which finds the roots of the HI profile at the FWHM, to set to W50 width. 
    Returns an array of all roots r[] found where the FWHM width is r[-1] - r[0] '''

    spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
    return spline.roots() 
    
def assign_units(e, B, W, peak_S=1):
    ''' Helper function which scales the unitless axes of the generalized busy function into meaningful km/s and mJy Spectrum units '''

    S = (B/np.max(B)) * peak_S       # Peak Flux set to 1 mJy if not specified
    r = find_FWHM(e, B)          
    FWHM = r[-1] - r[0]
    if W > 10:
        V = (e/FWHM)*(W)                 # set FWHM width to W50
    else:
        V = (e/FWHM)*(10)                # set FWHM width to W50
    W_ = (r/FWHM)*(W)                # Find the roots in units km/s
    return V, S, W_   # Return V and S axes and W_ roots (for checking)

def initial_guess(MHI, W50, D):
    return (MHI * 1000) / (W50 * 2.356e5 * (D**2))

# Gaussian Function
def normalDist(x, sigma, x0=0):
    return (1/(sigma*2*np.pi))*np.exp(-(x-x0)**2 / (2*sigma**2))

def Generate_Spectra(MHI, VHI, i, D, a=None, b1=None, b2=None, c=None, w=1, n=2, xe=0, xp=0):
    ''' Main function which generates an HI Spectrum. 
    The shape of the busy function (specified by a, b1, b2, c) is randomly generated (unless otherwise specified). 
    The area under the profile is set by MHI. 
    The FWHM width (W50) is set by VHI and inclination using W50 = VHI*2sin(i). 
    The profile is centered around 0 so xe=0 and xp=0. We use a second order n=2 general busy function'''

    W = VHI*2*np.sin(i) # W_50              
    e = np.linspace(-10, 10, 10000)  # unitless axes used in generalized busy function definition

    # Made faster by including initial guess for 'a' based on rough 'regular' integral -> S_peak * W50
    a_init = initial_guess(MHI, W, D)
    #print(a_init)

    #try_a = np.linspace(a_init-20, a_init+20, 1000)  # range of peak flux values to try around +/- 20 mJy of initial guess 
    if c is None: 
        c = np.random.choice(np.linspace(0, 1, 1000), 1)  # c controls the trough of the profile, choose randomly 
    if b1 is None:
        b1 = np.random.choice(np.linspace(1, 3, 1000), 1) # b1 controls the height of one peak in double profile, choose randomly
    if b2 is None:
        b2 = np.random.choice(np.linspace(1, 3, 1000), 1) # b2 controles the height of second peak in double profile, choose randomly
    
    # Find which peak value gives the correct integrated mass MHI:
    if a is None:
        try_M = 0
        a = a_init-20
        while try_M < MHI:
            B = Busy_general(x=e, a=1, b1=b1, b2=b2, xe=xe, xp=xp, c=c, w=w, n=n) 
            V, S, W_ = assign_units(e, B, W, peak_S=a)
            #print(S)
            try_M = get_MHI(V, S, D) # Check if you can just scale integral rather than re-integrating, it will re-introduce
            a = a + 0.5 # change increment to change accuracy vs speed
        # The try_M approximates M_HI to 2-3 significant figures -> improve this later

    # specificy a if known, and do not want to rerun HI spectra generation
    else:
        B = Busy_general(x=e, a=1, b1=b1, b2=b2, xe=xe, xp=xp, c=c, w=w, n=n)
        V, S, W_ = assign_units(e, B, W, peak_S=a)
        try_M = get_MHI(V, S, D)
        # print(S)
        # print(a)
        # print(try_a)

    # Add thermal broadening to Busy Function:
    FWHM = 10
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    G = normalDist(V, sigma)
    unitG = G / np.trapz(G, dx=diff(V))
    delta = (np.max(V) - np.min(V)) / V.shape
    S_broad = fftconvolve(S, unitG, mode='same') * delta 
    
    return try_M, V, S_broad, W, W_, a, b1, b2, c

#######################################################

# See Example_Run.py on how to use
