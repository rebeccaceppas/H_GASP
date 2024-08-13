# H-GASP
HI Galaxy Simulation Pipeline

End-to-end simulation pipeline for HI galaxy surveys focused on the performance of the Canadian Hydrogen Observatory and Radio Transient Detector (CHORD). Includes generating spectra of HI galaxies from mock catalogs and injecting them into input maps. We add the possibility to explore the effects of increasing the spectral resolution of the observations with an up-channelization algorithm. We include noise and calibration errors. For the specific use of this pipeline with the radiocosmology package (https://github.com/radiocosmology), we create a bash script to run the pipeline the whole way through to perform a mock observation.

# Example products
- Beam Transfer Matrices
- Synthesized beams
- Semi-realistics input map simulations (HI galaxies, synchrotron, cosmological 21 cm emission, etc)
- Up-channelization tools
- Pristine visibilities
- Visibilities with noise
- Dirty maps
- Full end-to-end simulations with a full HI mock catalog
- Full end-to-end simulations with an arbitrary number of sources at specific locations or following some desired distribution.

# How to use this code
This code is supposed to be used in combination with the radiocosmology packages. That implies two things
1. Before using any of the tools, the radio cosmology packages must be loaded on cedar. The most up to date version at this time can be loaded through:

```
module --force purge
module use /project/rrg-kmsmith/shared/chord_env/modules/modulefiles/
module load chord/chord_pipeline/2023.06
```

2. Because this will be used on cedar, installing packages is not trivial so instead you can clone this repository into your scratch directory and import the modules from there. 

For tasks that don't consume a lot of compute power or memory, the modules can be used in an interactive node on cedar. For example, the full_run.py script in the tutorial walks you through how to use most of the pipeline for a small test case (low number of dishes and of frequency channels). This entire script can be ran in an interactive node with no problem and in under 1 hour. Interactive nodes can also be used if, for example, you already have the beam transfer matrices and up-channelization response matrix computed. A template job script (pipeline.sh) is also provided if you have a larger job to submit.

The resources folder contains the template config files, sample calibration errors, and the mock HI galaxy catalogs created by Akanksha Bij.
