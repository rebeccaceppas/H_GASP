# H-GASP
HI Galaxy Simulation Pipeline

End-to-end simulation pipeline for HI galaxy surveys focused on the performance of the Canadian Hydrogen Observatory and Radio Transient Detector (CHORD). Includes generating spectra of HI galaxies from mock catalogs and injecting them into input maps. We add the possibility to explore the effects of increasing the spectral resolution of the observations with an up-channelization algorithm. We include noise and calibration errors. For the specific use of this pipeline with the radiocosmology package (https://github.com/radiocosmology), we create a bash script to run the pipeline the whole way through to perform a mock observation. We add examples of tools that can be used to recover the galaxies from the output product.

# Example products
- Beam Transfer Matrices
- Synthesized beams
- Semi-realistics input map simulations (HI galaxies, synchrotron, cosmological 21 cm emission, etc)
- Up-channelization tools
- Pristine visibilities
- Visibilities with noise and/or calibration errors
- Dirty maps
- Full end-to-end simulations with a full HI mock catalog
- Full end-to-end simulations with an arbitrary number of sources at specific locations or following some desired distribution.
