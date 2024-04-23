import numpy as np


class FreqState(object):
    """Process and store the frequency spec from the command line.
    Note: removed the options class method 
    -- I didn't ever use it and also couldn't find where ListOfType and click came from."""

    def __init__(self):

        # Set the CHIME band as the internal default
        self.freq = (800.0, 400.0, 1025)

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