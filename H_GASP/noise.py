# new additions to the draco noise module 
# https://github.com/radiocosmology/draco/blob/a66283b55bc27bc7c7baae9e41fd9a739a6c624e/draco/synthesis/noise.py

import numpy as np

from caput import config
from draco.core import io, containers, task
from caput.time import STELLAR_S
from draco.util import random
from drift.core import manager
import h5py

_default_bitgen = np.random.SFC64(seed=247479859775347473167578167923530262728)
_rng = np.random.Generator(_default_bitgen)

class GaussianNoise(task.SingleTask, random.RandomTask):
    """Add Gaussian distributed noise to a visibility dataset.

    Note that this is an approximation to the actual noise distribution good only
    when T_recv >> T_sky and delta_time * delta_freq >> 1.

    Attributes
    ----------
    ndays : float
        Multiplies the number of samples in each measurement.
    set_weights : bool
        Set the weights to the appropriate values.
    add_noise : bool
        Add Gaussian noise to the visibilities. By default this is True, but it may be
        desirable to only set the weights.
    recv_temp : bool
        The temperature of the noise to add.
    """

    # setting the defaults
    recv_temp = config.Property(proptype=float, default=50.0)
    ndays = config.Property(proptype=float, default=733.0)
    set_weights = config.Property(proptype=bool, default=True)
    add_noise = config.Property(proptype=bool, default=True)

    def setup(self, manager=None):
        """Set the telescope instance if a manager object is given.

        This is used to simulate noise for visibilities that are stacked
        over redundant baselines.

        Parameters
        ----------
        manager : manager.ProductManager, optional
            The telescope/manager used to set the `redundancy`. If not set,
            `redundancy` is derived from the data.
        """
        if manager is not None:
            self.telescope = io.get_telescope(manager)
        else:
            self.telescope = None

    def process(self, data):
        """Generate a noisy dataset.

        Parameters
        ----------
        data : :class:`containers.SiderealStream` or :class:`containers.TimeStream`
            The expected (i.e. noiseless) visibility dataset. Note the modification
            is done in place.
        norm : array
            The envelope used to normalize the up-channelized input map

        Returns
        -------
        data_noise : same as parameter `data`
            The sampled (i.e. noisy) visibility dataset.
        """
        data.redistribute("freq")

        visdata = data.vis[:]

        # Get the time interval
        if isinstance(data, containers.SiderealStream):
            dt = 240 * (data.ra[1] - data.ra[0]) * STELLAR_S
            ntime = len(data.ra)
        else:
            dt = data.time[1] - data.time[0]
            ntime = len(data.time)

        # TODO: this assumes uniform channels
        df = data.index_map["freq"]["width"][0] * 1e6
        nfreq = data.vis.shape[0]
        nprod = len(data.prodstack)
        ninput = len(data.index_map["input"])

        # Consider if this data is stacked over redundant baselines or not.
        if (self.telescope is not None) and (nprod == self.telescope.nbase):
            redundancy = self.telescope.redundancy
        elif nprod == ninput * (ninput + 1) / 2:
            redundancy = np.ones(nprod)
        else:
            raise ValueError("Unexpected number of products")
        
        # Calculate the number of samples, this is a 1D array for the prod axis.
        nsamp = int(self.ndays * dt * df) * redundancy
        std = self.recv_temp / np.sqrt(nsamp)

        if self.add_noise:

            # frequency independent noise

            noise = random.complex_normal(
                (nfreq, nprod, ntime),
                scale=std[np.newaxis, :, np.newaxis],
                rng=_rng,
            )

            # Iterate over the products to find the auto-correlations and add the noise
            for pi, prod in enumerate(data.prodstack):

                # Auto: multiply by sqrt(2) because auto has twice the variance
                if prod[0] == prod[1]:
                    visdata[:, pi].real += np.sqrt(2) * noise[:, pi].real

                else:
                    visdata[:, pi] += noise[:, pi]

        # Construct and set the correct weights in place
        if self.set_weights:
            for fi in range(len(visdata)):
                data.weight[fi] = 1.0 / std[:, np.newaxis] ** 2

        # Normalizing weights = normalizing N_inv for mapmaking
        data.weight[:] = data.weight[:]/np.max(data.weight[:])[0]

        return data

class NormalizedNoise(task.SingleTask, random.RandomTask):
    """Add Gaussian distributed noise to a visibility dataset that is renormalized by up-channelization.
    The noise will be channel dependent given by a normalization curve.

    Attributes
    ----------
    ndays: float
        Multiplies the number of samples in each measurement
        Number of days of observation
    set_weights: bool
        Set the visibility weights to the appropriate values
    add_noise: bool
        Add Gaussian noise to the visibilities. Default is True, but it may be
        desirable to set only weights.
    recv_temp: float
        The temperature of the noise to add
    """

    # setting the defaults
    recv_temp = config.Property(proptype=float, default=50.0)
    ndays = config.Property(proptype=float, default=733.0)
    set_weights = config.Property(proptype=bool, default=True)
    add_noise = config.Property(proptype=bool, default=True)

    def setup(self, manager=None):
        """Set the telescope instance if a manager object is given.

        This is used to simulate noise for visibilities that are stacked
        over redundant baselines.

        Parameters
        ----------
        manager : manager.ProductManager, optional
            The telescope/manager used to set the `redundancy`. If not set,
            `redundancy` is derived from the data.
        """
        if manager is not None:
            self.telescope = io.get_telescope(manager)
        else:
            self.telescope = None

    def process(self, data, norm):
        """Generate a noisy dataset.

        Parameters
        ----------
        data : :class:`containers.SiderealStream` or :class:`containers.TimeStream`
            The expected (i.e. noiseless) visibility dataset. Note the modification
            is done in place.
        norm : array
            The envelope used to normalize the up-channelized input map

        Returns
        -------
        data_noise : same as parameter `data`
            The sampled (i.e. noisy) visibility dataset.
        """
        data.redistribute("freq")

        visdata = data.vis[:]

        # Get the time interval
        if isinstance(data, containers.SiderealStream):
            dt = 240 * (data.ra[1] - data.ra[0]) * STELLAR_S
            ntime = len(data.ra)
        else:
            dt = data.time[1] - data.time[0]
            ntime = len(data.time)

        # TODO: this assumes uniform channels
        df = data.index_map["freq"]["width"][0] * 1e6
        nfreq = data.vis.shape[0]
        nprod = len(data.prodstack)
        ninput = len(data.index_map["input"])

        # Consider if this data is stacked over redundant baselines or not.
        if (self.telescope is not None) and (nprod == self.telescope.nbase):
            redundancy = self.telescope.redundancy
        elif nprod == ninput * (ninput + 1) / 2:
            redundancy = np.ones(nprod)
        else:
            raise ValueError("Unexpected number of products")
        
        # Calculate the number of samples, this is a 1D array for the prod axis.
        nsamp = int(self.ndays * dt * df) * redundancy
        std = self.recv_temp / np.sqrt(nsamp)

        if self.add_noise:

            # frequency independent noise
            noise_temp = random.complex_normal(
                (nfreq, nprod, ntime),
                scale=std[np.newaxis, :, np.newaxis],
                rng=_rng,
            )

            # normalizing by upchannelization envelope
            noise = np.copy(noise_temp)
            noise = noise / np.max(noise)  # care about relative amplitude change
            for i, n in enumerate(norm):
                noise[i,:,:] /= n
            
            # Iterate over the products to find the auto-correlations and add the noise
            for pi, prod in enumerate(data.prodstack):

                # Auto: multiply by sqrt(2) because auto has twice the variance
                if prod[0] == prod[1]:
                    visdata[:, pi].real += np.sqrt(2) * noise[:, pi].real

                else:
                    visdata[:, pi] += noise[:, pi]

        # Construct and set the correct weights in place
        if self.set_weights:
            for fi in range(len(visdata)):
                data.weight[fi] = 1.0 / std[:, np.newaxis] ** 2

        # Normalizing weights = normalizing N_inv for mapmaking
        data.weight[:] = data.weight[:]/np.max(data.weight[:])[0]

        return data


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