# Code developed with Olivia Pereira #

import numpy as np
from scipy import integrate

def draw_sample(x, y, n):
    ''' Sampling function to draw n samples from a distribution y(x) '''
    y_cumulative = integrate.cumtrapz(y, initial = 0)
    
    # randomly sampling based on original probability
    choices = np.random.uniform(low=0, high=1, size=n)
    draws = np.interp(choices, y_cumulative, x)
    return draws


################### wrapper functions for visibility calibration errors ###################

def draw_random_errors(phase_bins, phase_heights, amp_bins, amp_heights, output_shape):
    ''' 
    Draws phase and amplitude errors from given error distributions 
    
    Inputs
    - phase_bins, amp_bins: <array>
      x-values on which the error distributions for phase and amplitude are given
    - phase_heights, amp_heights: <array>
      y-values of the error distribution for phase and amplitude
    - output_shape: <tuple>
      desired shape of the output errors

    Outputs
    - phase_draws, amp_draws: <array>
      phase and amplitude errors sampled from the distribution
      shape is the requeste output_shape
    '''
    # normalizing the input heights to make it into a probability distribution
    phase_heights = phase_heights / integrate.trapz(np.abs(phase_heights))
    amp_heights = amp_heights / integrate.trapz(np.abs(amp_heights))
    
    # getting total random errors needed to populate the result matrix
    n = np.prod(output_shape)

    # getting phase errors for each matrix
    phase_draws = draw_sample(phase_bins, phase_heights, n).reshape(output_shape)
    amp_draws = draw_sample(amp_bins, amp_heights, n).reshape(output_shape)
    
    return phase_draws, amp_draws


def get_calibration_errors(amplitude_path, phase_path, output_shape):
    '''
    Loads the amplitude and phase error distributions from .npy files
    Draws the error samples and creates a complex gain factor that can be applied to 
    visibilities to introduce the desired phase errors and percent amplitude errors
    
    Inputs
    - amplitude_path, phase_path: <str>
      file paths for amplitude and phase error distributions
    - output_shape: <tuple>
      desired shape of the output gain factor

    Output
    - G: <array>
      complex error gain factor to apply to the visibilities
    '''


    amp = np.load(amplitude_path)
    phase = np.load(phase_path)
    percentages, amp_errors = amp[:,0], amp[:,1]
    radians, phase_errors = phase[:,0], phase[:,1]
    
    phase_draws, amp_draws = draw_random_errors(radians, phase_errors, percentages, amp_errors, output_shape)

    G = (1+amp_draws/100)*np.exp(1j*phase_draws)
    
    return G