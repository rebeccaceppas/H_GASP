# TO DO: figure out a scaling relationship for the job time, memory requirements

import numpy as np
from astropy import constants, units
import h5py
from draco.core import io, containers
from drift.core import manager


###### for opening and retriving info from pipeline generated files ######
def get_manager(output_folder):

    pm = manager.ProductManager.from_config(output_folder)

    return pm

def get_telescope(manager):
    
    telescope = io.get_telescope(manager)

    return telescope

def get_sstream(manager_folder, sstream_filename):

    pm = get_manager(manager_folder)
    tel = get_telescope(pm)

    data = h5py.File(sstream_filename)

    freqs = data['index_map']['freq'][:]
    freqmap = np.array([ii[0] for ii in freqs])
    vis = data['vis']
    weight = data['vis_weight']
    mmax = tel.mmax
    ntime = 2 * mmax + 1
    feed_index = tel.input_index

    kwargs = {}
    if tel.npairs != (tel.nfeed + 1) * tel.nfeed // 2:
        kwargs["prod"] = tel.index_map_prod
        kwargs["stack"] = tel.index_map_stack
        kwargs["reverse_map_stack"] = tel.reverse_map_stack

    else:
        # Construct a product map as if this was a down selection
        prod_map = np.zeros(
            tel.uniquepairs.shape[0], dtype=[("input_a", int), ("input_b", int)]
        )
        prod_map["input_a"] = tel.uniquepairs[:, 0]
        prod_map["input_b"] = tel.uniquepairs[:, 1]

        kwargs["prod"] = prod_map

    sstream = containers.SiderealStream(
    freq=freqs,
    ra=ntime,
    input=feed_index,
    distributed=True,
    **kwargs,
    )
    sstream.vis[:] = vis
    sstream.weight[:] = weight

    return sstream

###### make sure the directory inputs have correct format #####
def correct_directory(directory):
    '''Makes sure that the path will be completed by adding a / if missing.'''
    if directory[-1] != '/':
        return directory+'/'
    else:
        return directory

###### elevation of the pointing from zenith as required in radio cosmology pipeline ######
def get_elevation(pointing):
    '''Calculates the elevation relative to CHORD zenith in degrees.
       Positive is north of zenith, negative is south of zenith.'''
    zenith = 49.3207092194
    elevation = pointing - zenith
    return elevation

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


###### for estimating the amount of time and nodes needed for a job #####
def estimate_time_nodes(nfreq, Ndishes_ns, Ndishes_ew):
    '''
    estimates the time needed and the node configuration to request for the job
    given the number of frequencies and the array configuration

    Note: this is only an estimate and if it is not enough to complete your job can be adjusted manually.
    '''

    pass

