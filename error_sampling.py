import numpy as np
from scipy import integrate

# helper function
def draw_sample(bins, heights, n):
    # cumulatively integrating 
    y_cumulative = integrate.cumtrapz(heights, initial = 0)
    draws = []

    # randomly sampling based on original probability
    for i in range(n):
        choice = np.random.uniform(low = 0, high = 1)
        draw = np.interp(choice, y_cumulative, bins)
        draws.append(draw)

    return np.array(draws)


# wrapper function
def draw_random_errors(phase_bins, phase_heights, amp_bins, amp_heights, output_shape):

    # normalizing the input heights to make it into a probability distribution
    phase_heights = phase_heights / integrate.trapz(np.abs(phase_heights))
    amp_heights = amp_heights / integrate.trapz(np.abs(amp_heights))
    
    # getting total random errors needed to populate the result matrix
    n = np.prod(output_shape)

    # getting phase errors for each matrix
    phase_draws = draw_sample(phase_bins, phase_heights, n).reshape(output_shape)
    amp_draws = draw_sample(amp_bins, amp_heights, n).reshape(output_shape)
    
    return phase_draws, amp_draws


# wrapper function for once we have the file 
def get_calibration_errors(path_to_files, output_shape):

    amplitude_file = path_to_files + 'visibility_amplitude_errors.npy'
    phase_file = path_to_files + 'visibility_phase_errors.npy'

    amp = np.load(amplitude_file)
    phase = np.load(phase_file)
    percentages, amp_errors = amp[:,0], amp[:,1]
    radians, phase_errors = phase[:,0], phase[:,1]
    
    phase_draws, amp_draws = draw_random_errors(radians, phase_errors, percentages, amp_errors, output_shape)

    G = (1+amp_draws/100)*np.exp(1j*phase_draws)
    
    return G