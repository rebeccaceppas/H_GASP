"""
Classes and functions for mock observation steps

- beam transfer matrices
- up-channelization
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
import yaml
import os

class BeamTransferMatrices():

    def __init__(self, f_start, f_end, nfreq, output_directory,
                 CHORDdec_pointing=0, 
                 n_dishes_ew=11, n_dishes_ns=6, spacing_ew=6.3, spacing_ns=8.5, dish_diameter=6.0,
                 beam_spec='airy', Tsys=30, ndays=1, auto_correlation=False):
        
        '''defaults are for CHORD pathfinder pointing at zenith'''

        self.f_start = f_start
        self.f_end = f_end
        self.nfreq = nfreq

        self.elevation = get_elevation(CHORDdec_pointing)

        self.n_dishes_ew = n_dishes_ew
        self.n_dises_ns = n_dishes_ns
        self.spacing_ew = spacing_ew
        self.spacing_ns = spacing_ns
        
        self.dish_diameter = dish_diameter
        self.beam_spec = beam_spec
        self.Tsys = Tsys
        self.ndays = ndays
        self.auto_correlation = auto_correlation

        self.output_directory = output_directory
    
    def change_config(self):
        '''updates the template config file beam.yaml to represent the desired instrument.
           creates a new file beam.yaml in the output directory specified.'''

        with open('beam.yaml') as istream:
            ymldoc = yaml.safe_load(istream)
            ymldoc['telescope']['freq_start'] = float(self.f_start)
            ymldoc['telescope']['freq_end'] = float(self.f_end)
            ymldoc['telescope']['num_freq'] = self.nfreq
            ymldoc['telescope']['elevation_start'] = self.elevation
            ymldoc['telescope']['elevation_end'] = self.elevation
            ymldoc['telescope']['ndays'] = self.ndays
            ymldoc['telescope']['layout_spec']['grid_ew'] = self.n_dishes_ew
            ymldoc['telescope']['layout_spec']['grid_ns'] = self.n_dises_ns
            ymldoc['telescope']['layout_spec']['spacing_ew'] = self.spacing_ew
            ymldoc['telescope']['layout_spec']['spacing_ns'] = self.spacing_ns
            ymldoc['telescope']['beam_spec']['diameter'] = self.dish_diameter
            ymldoc['telescope']['beam_spec']['type'] = self.beam_spec
            ymldoc['telescope']['tsys_flat'] = float(self.Tsys)
            ymldoc['telescope']['auto_correlation'] = self.auto_correlation
            ymldoc['config']['output_directory'] = self.output_directory
        istream.close()

        with open(self.output_directory+'/beam.yaml', 'w') as ostream:
            yaml.dump(ymldoc, ostream, default_flow_style=False, sort_keys=False)

    def get_beam_transfer_matrices(self):

        '''submits the job for simulating the btm with drift makeproducts'''

        self.change_config()

        package_path= '/project/6002277/ssiegel/chord/chord_env/modules/chord/chord_pipeline/2022.11/lib/python3.10/site-packages/drift/scripts/makeproducts.py '
        action = 'run {}/beam.yaml '.format(self.output_directory)
        log = '&> {}/beam.log'.format(self.output_directory)

        command = package_path + action + log

        os.system('srun python ' + command)


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

    def __init__(self, output_directory, btm_directory, maps_tag, map_filepaths=[]) -> None:
        
        self.output_directory = output_directory
        self.btm_directory = btm_directory
        self.map_filepaths = map_filepaths
        self.maps_tag = maps_tag
        self.output_filename = self.output_directory+'/sstream_{tag}.h5'
        self.n_maps = len(map_filepaths)

    def change_config(self):

        with open("simulate.yaml") as istream:
            ymldoc = yaml.safe_load(istream)
            ymldoc['cluster']['directory'] = self.output_directory+'/visibilities_info'
            ymldoc['pipeline']['tasks'][1]['params']['product_directory'] = self.btm_directory
            
            # sidereal sstream
            ymldoc['pipeline']['tasks'][3]['params']['output_name'] = self.output_filename
            ymldoc['pipeline']['tasks'][3]['params']['tag'] = self.maps_tag
            
            # load map tag
            ymldoc['pipeline']['tasks'][2]['params']['maps'][0]['tag'] = self.maps_tag

            # add the data for the maps
            ymldoc['pipeline']['tasks'][2]['params']['maps'][0]['files'][0] = self.map_filepaths[0]
            
            if self.n_maps > 1:
                for file in self.map_filepaths[1:]:
                    ymldoc['pipeline']['tasks'][2]['params']['maps'][0]['files'].append(file)
                                           
        istream.close()

        with open(self.output_directory+"/simulate.yaml", "w") as ostream:
            yaml.dump(ymldoc, ostream, default_flow_style=False, sort_keys=False)
        ostream.close()

    def get_visibilities(self):

        '''submits the job for simulating the btm with caput pipeline'''

        self.change_config()

        package_path = '/project/6002277/ssiegel/chord/chord_env/modules/chord/chord_pipeline/2022.11/lib/python3.10/site-packages/caput/scripts/runner.py '
        action = 'run {}/simulate.yaml '.format(self.output_directory)
        log = '&> {}/visibilities.log'.format(self.output_directory)

        command = package_path + action + log
        os.system('srun python ' + command)


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

def get_elevation(pointing):
    '''Calculates the elevation relative to CHORD zenith in degrees.
       Positive is north of zenith, negative is south of zenith.'''
    zenith = 49.3207092194
    elevation = pointing - zenith
    return elevation