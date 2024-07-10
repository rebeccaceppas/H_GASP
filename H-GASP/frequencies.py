import numpy as np
import channelization_functions as cf

class FreqState(object):
    ## Code snipet modified from radiocosmology package https://github.com/radiocosmology ##
    """Process and store frequency properties."""

    def __init__(self):

        # Set the CHORD band as the internal default
        self.freq = (1500.0, 300.0, 2048)

        self.channel_range = None
        self.channel_list = None
        self.channel_bin = 1
        self.freq_mode = "centre"

    @property
    def frequencies(self):
        """The frequency centres in MHz."""
        return self._calculate()[0]

    @property
    def freq_width(self):
        """The frequency width in MHz."""
        return self._calculate()[1]

    def _calculate(self):
        """Calculate the frequencies from the parameters."""
        # Generate the set of frequency channels given the parameters

        sf, ef, nf = self.freq
        if self.freq_mode == "centre":
            df = abs(ef - sf) / nf
            frequencies = np.linspace(sf, ef, nf, endpoint=False)
        elif self.freq_mode == "centre_nyquist":
            df = abs((ef - sf) / (nf - 1))
            frequencies = np.linspace(sf, ef, nf, endpoint=True)
        else:
            df = (ef - sf) / nf
            frequencies = sf + df * (np.arange(nf) + 0.5)

        # Rebin frequencies if needed
        if self.channel_bin > 1:
            frequencies = frequencies.reshape(-1, self.channel_bin).mean(axis=1)
            df = df * self.channel_bin

        # Select a subset of channels if required
        if self.channel_list is not None:
            frequencies = frequencies[self.channel_list]
        elif self.channel_range is not None:
            frequencies = frequencies[self.channel_range[0] : self.channel_range[1]]

        return frequencies, df

    @classmethod
    def _set_attr(cls, ctx, param, value):
        state = ctx.ensure_object(cls)
        setattr(state, param.name, value)
        return value


def get_frequencies(fmax_obs, fmin_obs, U, fmin=300, number_channels=2048, sampling_rate=0.417):
    '''
    Calculates the exact frequencies of the observation
    given a maximum and minimum frequencies and up-channelization factor U.

    Useful for when  making maps and simulating beam transfer matrices.

    Inputs
    ------
    - fmax_obs: <float>
      maximum frequency to consider for this specific observation in MHz
    - fmin_obs: <float>
      minimum frequency to consider for this specific observation in MHz
    - fmin: <float>
      minimum instrumental frequency. default is CHORD's 300 MHz
    - number_channels: <int>
      number of channels. default is CHORD's 2048
    - sampling_rate: <float>
      time stream sampling rate in ns. default is CHORD's 0.417 ns  

    Outputs
    -------
    - fstate: <FreqState object>
      object containing the frequency specifications used throughout the pipeline
    - f_start: <float>
      frequency of the initial channel**
    - f_end: <float>
      frequency of the final channel**
    - nfreq: <int>
      number of frequency channels in up-channelized observation**

    ** these are the quatities to be fed into the map-making for input maps and the 
       beam transfer matrix computation using drift.
       '''

    chans = cf.get_chans(fmax_obs, fmin_obs,
                         fmin=fmin,
                         number_channels=number_channels,
                         sampling_rate=sampling_rate)
    
    use_max = cf.freq_unit_add(np.arange(chans.min()-0.5 + 1/(2*U), chans.max()+0.5, 1/U).max(),
                               fmin=fmin,
                               number_channels=number_channels,
                               sampling_rate=sampling_rate)
    
    use_min = cf.freq_unit_add(np.arange(chans.min()-0.5 + 1/(2*U), chans.max()+0.5, 1/U).min(),
                               fmin=fmin,
                               number_channels=number_channels,
                               sampling_rate=sampling_rate)
    
    df = (use_max - use_min) / (chans.size*U -1)
    size_freqs = chans.size*U

    f_start = use_max
    f_end = use_min - df
    nfreq = int(size_freqs)

    fstate = FreqState()
    fstate.freq = (f_start, f_end, nfreq)

    return fstate, f_start, f_end, nfreq