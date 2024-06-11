"""
Classes and functions for mock observation steps

- up-channelization
- visibilities
- noisy visibilities
- visibilities with calibration errors
- dirty maps
"""

import channelization_functions as cf

class UpchannelizedMap():

    def __init__(self, U) -> None:
        
        self.U = U



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