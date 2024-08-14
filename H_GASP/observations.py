"""
Classes and functions for mock observation steps

- beam transfer matrices
- up-channelization
- visibilities
- noisy visibilities
- dirty maps
"""

from H_GASP import channelization_functions as cf
from H_GASP import utilities
from H_GASP import frequencies as fr
from H_GASP import savetools
import h5py
import numpy as np
import yaml
import os


class BeamTransferMatrices():
    '''Class to hold parameters and methods to simulate beam transfer matrices (BTM)
       Defaults are for CHORD pathfinder pointing at zenith'''
    def __init__(self, f_start, f_end, nfreq, output_directory, H_GASP_path,
                 CHORDdec_pointing=49.3207092194, 
                 n_dishes_ew=11, n_dishes_ns=6, spacing_ew=6.3, spacing_ns=8.5, dish_diameter=6.0,
                 beam_spec='airy', Tsys=30, ndays=1, auto_correlation=True):
        
        '''
        - f_start, f_end: <float>
          The starting and ending frequency for which to perform the simulation.
          Can be set by the user if there are pre-made maps.
          If up-channelization is being used, these will be output by the upchannelize method.
        - nfreq: <int>
          Number of frequency channels to do this simulation for.
          Can be set by user with a minimum of nfreq = 2.
          If up-channelization is being used, this will be output by the upchannelize method.
        - output_directory: <str>
          Path and name of directory under which to store the BTM
        - H_GASP_path: <str>
          Path to where H_GASP has been cloned. Required for retrieving the config files.
        - CHORDdec_pointing: <float>
          Desired Declination to which we point CHORD in degrees.
          Default is zenith.
        - n_dishes_ew, n_dishes_ns: <int>
          Number of dishes in the East-West and North-South directions.
        - spacing_ew, spacing_ns: <float>
          Spacing between the center of the dishes in the East-West and North-South directions.
          The values should be in meters.
        - dish_diameter: <float>
          The diameter of each individual dish in meters.
        - beam_spec: <str>
          The particular shape of the primary beam of each dish.
          Options supported by the CHORD pipeline are 'airy', 'gaussian', or 'healpix'.
          'healpix' is a HEALPixBeam object input by user
        - Tsys: <float>
          System temperature in K
        - ndays: <float>
          number of days of observation, for noise estimation purposes
        - auto_correlation: <bool>
          Whether or not to include the auto-correlation in the computations
        '''

        self.f_start = f_start
        self.f_end = f_end
        self.nfreq = nfreq

        self.elevation = utilities.get_elevation(CHORDdec_pointing)

        self.n_dishes_ew = n_dishes_ew
        self.n_dises_ns = n_dishes_ns
        self.spacing_ew = spacing_ew
        self.spacing_ns = spacing_ns
        
        self.dish_diameter = dish_diameter
        self.beam_spec = beam_spec
        self.Tsys = Tsys
        self.ndays = ndays
        self.auto_correlation = auto_correlation

        self.output_directory = utilities.correct_directory(output_directory)
        self.H_GASP = utilities.correct_directory(H_GASP_path)
    
    def change_config(self):
        '''updates the template config file beam.yaml to represent the desired instrument.
           creates a new file beam.yaml in the output directory specified.'''
        
        with open(self.H_GASP + 'resources/beam.yaml') as istream:
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

        '''runs the job for simulating the btm with drift makeproducts'''

        self.change_config()

        package_path= '/project/6002277/ssiegel/chord/chord_env/modules/chord/chord_pipeline/2022.11/lib/python3.10/site-packages/drift/scripts/makeproducts.py '
        action = 'run {}beam.yaml '.format(self.output_directory)
        log = '&> {}beam.log'.format(self.output_directory)

        command = package_path + action + log

        os.system('srun python ' + command)


class Upchannelization():
    '''Class for up-channelizing maps or galaxy catalogs and calcilating important frequencies'''

    def __init__(self, U, fmax, fmin, output_directory, output_filename, R_filename, norm_filename, freqs_matrix_filename):
        '''
        - U: <int>
          Up-channelization factor, U = 2^n
        - fmax, fmin: <float>
          Maximum and minimum frequencies of interest.
          The max/min frequencies of the output will be given by the nearest calculated fine channel.
        - output_directory: <str>
          The path and directory onto which we save all the products
        - output_filename: <str>
          The name with which to save the output file 
        - R_filename: <str>
          The file name for the response matrix R. If it has already been simulated this will be
          used to retrieve it, otherwise this will be the name given to the new file.
        - norm_filename: <str>
          The file name for the normalization vector. If it has already been simulated this will be
          used to retrieve it, otherwise this will be the name given to the new file.
        - freqs_matrix_filename: <str>
          The file name for the output frequencies of the up-channelization process.
          If it has already been simulated this will be used to retrieve it, 
          otherwise this will be the name given to the new file.
        '''
        
        self.U = U
        self.fmax = fmax
        self.fmin = fmin
        self.output_name = output_filename
        self.output_directory = utilities.correct_directory(output_directory)
        self.R_filename = R_filename
        self.norm_filename = norm_filename
        self.freqs_matrix_filename = freqs_matrix_filename
        

    def get_R_norm(self):
        '''Computes the response matrix R and the normalization vector to remove the modulations
           This method shoud only be called if these have not yet been computed.'''
        
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
        '''
        Up-channelizes the input map or the input catalog to the specifications given.

        Inputs:
        ------
        - catalog: <bool>
          If set to True, a catalog file path must be provided and an up-channelized map will be created from it.
          If set to False (default), a map file path must be provided and the algorithm will up-channelize that map.
        - map_paths: <str> or <list of str>
          Contains the paths and names of all the maps to up-channelize.
          If there are multiple maps, they must be given in a list, otherwise a string.
          A file called full_input.h5 will be created with the sum of all the provided maps.
        - catalog_filepath: <str>
          If catalog=True this is the file path to the HI mock catalog from which to create an up-channelized map.
          This is prefered over first creating a map and then up-channelizing if no other components are needed
          as it will save memory and prevent information of profile shapes from being lost prematurely.
        - nside: <int>
          Sets the resolution of the output map. nside = 2^n.
        - b_max: <float>
          Maximum baseline length in your given dish array configuration in metres.
          The default is for the CHORD pathfinder, to estimate a new one use H_GASP.utilities.calculate_baseline_lengths

        Outputs:
        -------
        - creates the up-channelized map and saves to disk
        - f_start, f_end: <float>
          Max and min frequencies of the up-channelized map corresponding to the fine channels.
          These exact values should be used when computing the beam transfer matrices.
        - nfreq: <int>
          Number of fine frequency channels.
          This exact value should be used when computing the beam transfer matrices.
        '''
        fstate, f_start, f_end, nfreq = fr.get_frequencies(self.fmax, self.fmin, self.U)
        fine_freqs = cf.get_fine_freqs(fstate.frequencies)

        print('The exact frequency specifications of this observation are:')
        print('f_start = {}, f_end = {}, nfreq = {}.'.format(f_start, f_end, nfreq))
        print('Use these exact values when computing the beam transfer matrices with drift. They are returned by this function.')

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
            # open the input maps to up-channelize
            # combine all maps into 1
            open_maps(map_paths,
                  self.output_directory + 'full_input.h5')
            
            cf.channelize_map(self.U,
                              fstate,
                              self.output_directory + 'full_input.h5',
                            self.output_directory + self.R_filename,
                            self.output_directory + self.norm_filename,
                            self.output_directory + self.freqs_matrix_filename,
                            fine_freqs,
                            self.output_directory + self.output_name)
            
            print('Channelized map saved at', self.output_directory + self.output_name)

        return f_start, f_end, nfreq


class Visibilities():
    '''Class for simulating noiseless visibilities from input sky maps'''

    def __init__(self, output_directory, btm_directory, H_GASP_path, maps_tag, map_filepaths=[]):
        '''
        - output_directory: <str>
          The path and directory onto which we save all the products
        - btm_directory: <str>
          Path to directory containing the beam transfer matrices
        - H_GASP_path: <str>
          Path to where H_GASP has been cloned. Required for retrieving the config files.
        - maps_tag: <str>
          A tag to add to the file name to remember context, for example which components were in the input maps
        - map_filepaths: <list of str>
          Paths and names of all maps to be included in the final visibility simulation.
          If there is only one map, it should still be given as a str in a list 
        '''

        self.output_directory = utilities.correct_directory(output_directory)
        self.btm_directory = utilities.correct_directory(btm_directory)
        self.H_GASP = utilities.correct_directory(H_GASP_path)

        self.map_filepaths = map_filepaths
        self.maps_tag = maps_tag
        self.output_filename = self.output_directory+'sstream_{tag}.h5'
        self.n_maps = len(map_filepaths)

    def change_config(self):
        '''updates the template config file simulate.yaml to perform the visibility task.
           creates a new file simulate.yaml in the output directory specified.'''

        with open(self.H_GASP + 'resources/simulate.yaml') as istream:
            ymldoc = yaml.safe_load(istream)
            ymldoc['cluster']['directory'] = self.output_directory+'visibilities_info'
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

        with open(self.output_directory+"simulate.yaml", "w") as ostream:
            yaml.dump(ymldoc, ostream, default_flow_style=False, sort_keys=False)
        ostream.close()

    def get_visibilities(self):
        '''
        Runs the job for simulating the visibilities with caput pipeline

        Outputs:
        -------
        - data: <sstream container>
          the visibility data container that is needed to simulate dirty maps.
        '''
        self.change_config()

        package_path = '/project/6002277/ssiegel/chord/chord_env/modules/chord/chord_pipeline/2022.11/lib/python3.10/site-packages/caput/scripts/runner.py '
        action = 'run {}simulate.yaml '.format(self.output_directory)
        log = '&> {}visibilities.log'.format(self.output_directory)

        command = package_path + action + log
        os.system('srun python ' + command)

        sstream_file = utilities.correct_directory(self.output_directory) + 'sstream_{}.h5'.format(self.maps_tag)
        data = utilities.get_sstream(self.btm_directory, sstream_file)

        return data


class RealisticVisibilities():

    '''Class for adding noise to pristine visibilities.'''

    def __init__(self, ndays, btm_directory, output_directory, maps_tag='', noiseless_vis_filename='', Tsys=30):
        '''
        - ndays: <float>
          number of days of observation
        - btm_directory: <str>
          Path to directory containing the beam transfer matrices
        - output_directory: <str>
          The path and directory onto which we save all the products
        - maps_tag: <str>
          A tag to add to the file name to remember context, for example which components were in the input maps.
          This tag was passed on to the Visibilities() object
        - Tsys: <float>
          System temperature in K. Default is 30 K.
        '''
        self.manager = utilities.get_manager(btm_directory)
        self.ndays = ndays
        self.tsys = Tsys

        if noiseless_vis_filename == '':
          sstream_file = utilities.correct_directory(output_directory) + 'sstream_{}.h5'.format(maps_tag)

        else:
            sstream_file = utilities.correct_directory(output_directory) + noiseless_vis_filename

        self.data = utilities.get_sstream(btm_directory, sstream_file)
        
    def add_noise(self, upchannelized=True, norm_filepath=''):
        '''
        Adds noise to noiseless visibilities. Adjusts the noise if the map has been up-channelized
        to account for diving out the coarse channel-scale ripples.

        Inputs:
        ------
        - upchannelized: <bool>
          If True (default) will renormalize the noise variance so that there is a frequency dependent variance
          from which the Gaussian noise is drawn from for each channel.
          If False, the noise will be standard Gaussian noise with the same variance at all channels.
        - norm_filepath: <str>
          Path to the file containing the normalization vector.
          Only required if upchannelized is set to True

        Outputs:
        -------
        - noisy_data: <sstream container>
          the noisy visibility data container that is needed to simulate dirty maps.
        '''

        from H_GASP import noise

        dict_stream = {'recv_temp': self.tsys, 
               'ndays': self.ndays}

        if upchannelized:
            norm = np.load(norm_filepath)

            noisy = noise.NormalizedNoise()
            noisy.setup(self.manager)
            noisy.read_config(dict_stream)
            noisy_data = noisy.process(self.data, norm)

        else:
            noisy = noise.GaussianNoise()
            noisy.setup(self.manager)
            noisy.read_config(dict_stream)
            noisy_data = noisy.process(self.data)

        return noisy_data
                    

class DirtyMap():

    def __init__(self, data, output_filepath, fstate, nside, btm_directory, auto_correlation=True):
        '''
        - data: <sstream container>
          The visibility data as output from Visibilities(), RealisticVisibilities().
          It can also be obtained for a pre-computed visibility matrix with H_GASP.utilities.get_sstream()
        - output_filepath: <str>
          Path and name to which we save the computed dirty map
        - fstate: <FreqState object>
          From H_GASP.frequencies.FreqState(), contains information about the frequency specifications
          This is required to agree with the setting of your beam transfer matrices, so
          fstate = FreqState()
          fstate.freq = (f_start, f_end, nfreq)
        - nside: <int>
          Sets the resolution of the output map. nside = 2^n.
        - btm_directory: <str>
          Path to directory containing the beam transfer matrices
        - auto_correlation: <bool>
          Whether or not to include the auto-correlation in the computations
        '''

        self.btm_directory = btm_directory

        self.manager = utilities.get_manager(btm_directory)
        self.data = data

        self.dict_mask = {'auto_correlations': auto_correlation}
        self.dict_map = {'nside': nside}

        self.output_filepath = output_filepath

        self.fstate = fstate

    def compute_mmodes(self):
        '''Helper method to compute the m-modes using draco.analysis'''
        from draco.analysis import transform, flagging

        mmodes = transform.MModeTransform()
        mmodes.setup(self.manager)
        Mmodes = mmodes.process(self.data)

        # if auto_correlation was set to False this will mask it out
        # otherwise it will stay the same
        mmodes_masked = flagging.MaskMModeData()
        mmodes_masked.read_config(self.dict_mask)
        Mmodes_masked = mmodes_masked.process(Mmodes)

        self.mmodes = Mmodes_masked

    def get_dirty_map(self):
        '''Computes and saves dirty map to disk using draco.analysis'''
        from draco.analysis import mapmaker

        self.compute_mmodes()

        dm = mapmaker.DirtyMapMaker()
        dm.read_config(self.dict_map)
        dm.setup(self.manager)
        m = dm.process(self.mmodes)

        map_ = m['map'][:]
        savetools.write_map(self.output_filepath, 
                  map_, self.fstate.frequencies, self.fstate.freq_width, 
                  include_pol=True)



def load_map(map_path):
    '''helper function to load a single map'''
    f = h5py.File(map_path)
    sky_map = np.array(f['map'])
    ff = np.array(f['index_map']['freq'])
    freqs = np.array([ii[0] for ii in ff])
    f_width = np.abs(freqs[0] - freqs[1])
    f.close()

    return freqs, f_width, sky_map

def open_maps(map_paths, output_name):
    '''helper function to load multiple files
       saves a new file contianing the sum of all provided maps to disk'''
    if type(map_paths) == str:
        freqs, f_width, sky_map = load_map(map_paths)

    elif type(map_paths) == list:
        freqs, f_width, sky_map = load_map(map_paths[0])

        for i in range(1,len(map_paths)-1):
            print(i)
            freqs, f_width, new_map = load_map(map_paths[i])
            sky_map += new_map

    print('Writing file {} containing the input maps {}.'.format(output_name,
                                                                 map_paths))

    savetools.write_map(output_name,
                sky_map,
                freqs,
                f_width)
    
