"""
Classes and functions for mock observation steps

- up-channelization
- beam transfer matrices
- visibilities
- noisy visibilities
- visibilities with calibration errors
- dirty maps
"""

import channelization_functions as cf
import h5py
import numpy as np
import frequencies as fr
from savetools import write_map

class Upchannelization():

    def __init__(self, U, fmax, fmin, output_directory, output_filename, R_filename, norm_filename, freqs_matrix_filename) -> None:
        
        self.U = U
        self.fmax = fmax
        self.fmin = fmin
        self.output_name = output_filename
        self.output_directory = output_directory
        self.R_filename = R_filename
        self.norm_filename = norm_filename
        self.freqs_matrix_filename = freqs_matrix_filename
        

    def get_R_norm(self):
        # if the R and norm matrices have not already been computed, do it here
        fstate, f_start, f_end, nfreq = fr.get_frequencies(self.fmax, self.fmin, self.U)
        
        fine_freqs = cf.get_fine_freqs(fstate.frequencies)

        R, freqs_matrix, norm = cf.get_response_matrix(fine_freqs,
                                                self.U,
                                                self.fmin,
                                                self.fmax)
        
        try:
            np.save(self.output_directory+self.R_filename, R)
            np.save(self.output_directory+self.norm_filename, norm)
            np.save(self.output_directory+self.freqs_matrix_filename, freqs_matrix)
        except:
            print(self.output_directory, 'does not exist. Please create it and try again.')

        print('Up-channelization matrix with shape {} saved to {}.'.format(R.shape,
                                                                           self.output_directory+self.R_filename))
        print('Normalization vector with shape {} saved to {}.'.format(norm.shape,
                                                                           self.output_directory+self.norm_filename))
        print('Matrix frequencies with shape {} saved to {}'.format(freqs_matrix.shape,
                                                                    self.output_directory + self.freqs_matrix_filename))
        
    def upchannelize(self, catalog=False, map_paths='', catalog_filepath='', nside=128, b_max=77):

        fstate, f_start, f_end, nfreq = fr.get_frequencies(self.fmax, self.fmin, self.U)
        fine_freqs = cf.get_fine_freqs(fstate.frequencies)

        print('The exact frequency specifications of this observation are:')
        print('f_start = {}, f_end = {}, nfreq = {}.'.format(f_start, f_end, nfreq))
        print('Use these exact values when computing the beam transfer matrices with drift.')

        if catalog:

            print('Creating up-channelized map from catalog {} with nside = {}.'.format(catalog_filepath, nside))

            cf.channelize_catalogue(self.U,
                                    fstate,
                                    nside,
                                    catalog_filepath,
                                    self.output_directory + self.R_filename,
                                    self.output_directory + self.norm_filename,
                                    self.output_directory + self.freqs_matrix_filename,
                                    fine_freqs,
                                    self.output_directory + self.output_name,
                                    b_max = b_max)
            
            print('Channelized map saved at', self.output_directory + self.output_name)

        else:
            # open the input maps tp up-channelize
            # combine all maps into 1
            open_maps(map_paths,
                  self.output_directory + 'full_input.h5')
            
            cf.channelize_map(self.U,
                              self.output_directory + 'full_input.h5',
                            self.output_directory + self.R_filename,
                            self.output_directory + self.norm_filename,
                            self.output_directory + self.freqs_matrix_filename,
                            fine_freqs,
                            self.output_directory + self.output_name)
            
            print('Channelized map saved at', self.output_directory + self.output_name)


class Visibilities():

    def __init__(self, map_path) -> None:
        
        self.map = map_path

    def get_visibilities(self):

        # code to get visibilities
        pass

    def _add_noise(self):
        pass

    def _add_calibration_errors(self):
        pass


class NoisyVisibilities(Visibilities):

    def __init__(self, map_path) -> None:
        super().__init__(map_path)

    def _add_noise(self):
        return super()._add_noise()
    

class CalibrationErrorVisibilities(Visibilities):

    def __init__(self, map_path) -> None:
        super().__init__(map_path)

    def _add_calibration_errors(self):
        return super()._get_visibilities()


class RealisticVisibilities(Visibilities):

    def _add_noise(self):
        return super()._add_noise()

    def _add_calibration_errors(self):
        return super()._add_calibration_errors()    


class DirtyMap():

    def __init__(self, nside, beam_path, auto_correlations=False) -> None:

        self.nside = nside
        self.auto_correlations = auto_correlations
        self.beam_path = beam_path

    
    def compute_mmodes(self, visibilities_path):



        pass

    def compute_dirty_mao(self):

        pass



def load_map(map_path):

    f = h5py.File(map_path)
    sky_map = np.array(f['map'])
    ff = np.array(f['index_map']['freq'])
    freqs = np.array([ii[0] for ii in ff])
    f_width = np.abs(freqs[0] - freqs[1])
    f.close()

    return freqs, f_width, sky_map

def open_maps(map_paths, output_name):

    if type(map_paths) == str:
        freqs, f_width, sky_map = load_map(map_paths)

    elif type(map_paths) == list:
        freqs, f_width, sky_map = load_map(map_paths[0])

        for i in range(1,len(map_paths)-1):
            freqs, f_width, new_map = load_map(map_paths[i])
            sky_map += new_map

    print('Writing file {} containing the input maps {}.'.format(output_name,
                                                                 map_paths))

    write_map(output_name,
                sky_map,
                freqs,
                f_width)


def _write_map(map_filepath, sky_map, freqs, f_width, include_pol=True):
    # maybe have it here but probably worth it to have it in savetools.py 
    # or another file for this sort of utilities
    pass