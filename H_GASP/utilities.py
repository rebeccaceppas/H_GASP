# TO DO: figure out a scaling relationship for the job time, memory requirements

import numpy as np
from astropy import constants, units
from pathlib import Path
import os


###### find location of the H_GASP directory to locate the files and modules needed #####

def find_h_gasp_directory():
    start_search_path = Path.home()
    print('start', start_search_path)
    for root, dirs, files in os.walk(start_search_path):
        #print('dir', dirs)
        if 'H_GASP' in dirs:
            print('found', Path(root) / 'H_GASP')
            return Path(root) / 'H_GASP'
    return None

###### for beam width estimates ########

def calculate_baseline_lengths(Ndishes_ns, Ndishes_ew, spacing_ns, spacing_ew):
    '''calculates all baseline lenghts in the given array configuration
       returns a list of all lengths and the maximum length'''
    # Create arrays for the positions of the dishes
    ns_positions = np.arange(Ndishes_ns) * spacing_ns
    ew_positions = np.arange(Ndishes_ew) * spacing_ew
    
    # Create a grid of dish positions
    grid_x, grid_y = np.meshgrid(ns_positions, ew_positions)
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Calculate all pairwise distances
    num_dishes = Ndishes_ns * Ndishes_ew
    baseline_lengths = []
    for i in range(num_dishes):
        for j in range(i + 1, num_dishes):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = np.sqrt(dx**2 + dy**2)
            baseline_lengths.append(distance)
    
    return baseline_lengths, np.max(baseline_lengths)

def estimate_primary_beam_width(dish_diameter, frequency):
    '''estimates the primary beam width at a given frequency for some dish diameter'''
    wavelength = (constants.c / (frequency*units.MHz)).to(units.m)
    beam_width = wavelength / (dish_diameter*units.m)
    return (beam_width * units.rad).to(units.deg)

def estimate_synthesized_beam_width(Ndishes_ns, Ndishes_ew, spacing_ns, spacing_ew, frequency):
    '''estimates the synthesized beam width at a given frequency for some array configuration'''
    wavelength = (constants.c / (frequency*units.MHz)).to(units.m) 
    _, max_baseline = calculate_baseline_lengths(Ndishes_ns, Ndishes_ew, spacing_ns, spacing_ew)
    beam_width = wavelength / (max_baseline*units.m)
    return (beam_width * units.rad).to(units.deg)


###### for estimating the amount of time and nodes needed for a job ######

def estimate_time_nodes(nfreq, Ndishes_ns, Ndishes_ew):
    '''
    estimates the time needed and the node configuration to request for the job
    given the number of frequencies and the array configuration

    Note: this is only an estimate and if it is not enough to complete your job can be adjusted manually.
    '''

    pass

