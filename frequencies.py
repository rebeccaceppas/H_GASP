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


def get_frequencies(fmax, fmin, U):

    chans = cf.get_chans(fmax, fmin)
    print(chans)
    use_max = cf.freq_unit_add(np.arange(chans.min()-0.5 + 1/(2*U), chans.max()+0.5, 1/U).max())
    use_min = cf.freq_unit_add(np.arange(chans.min()-0.5 + 1/(2*U), chans.max()+0.5, 1/U).min())
    df = (use_max - use_min) / (chans.size*U -1)
    size_freqs = chans.size*U

    f_start = use_max
    f_end = use_min - df
    nfreq = int(size_freqs)

    print(f_start, f_end, nfreq)

    fstate = FreqState()
    fstate.freq = (f_start, f_end, nfreq)

    return fstate