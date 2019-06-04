#################
Development ideas
#################
Some ideas for future development of MDT.

- MPT (Microstructure Prototol Toolbox)
    - Perhaps a different package using MDT and MOT for protocol optimization
- More optimization routines:
    - L-BFGS-B
    - BOBYQA or NEWUOA from Powell
    - Particle Swarm
    - See the book "Introduction to derivative free optimization" or similar books
- More sampling routines:
    - emcee
- Automatic derivatives
    - This is only doable with OpenCL 2.2
- Python 3.7:
    - removed ordereddict
    - use f-strings
- OpenCL 2.0:
    - remove priors and merge with the constraints
- remove the JohnsonNoise model
    - have the OffsetGaussian an separate input for the offset
- protocol options using a selection matrix.
    - i.e. per protocol row a 0 or 1 if that protocol is supposed to be used
    - Allows protocol options per voxel.
- remove mot_float_type, replace with either float or double depending on insight/tests
- move g to gx, gy, gz
- remove the pre_transform_code from the dependencies (was used for the weights transformation, but no longer neeed)
- to discuss: add 'scale_factor' to the parameters as a simple scaling factor
    - for use in MLE and FIM
    - replacing ScaleTransform and scale_factor in numdiff_info
    - perhaps add it to MCMC as well
- Move to using an OpenCL Context instead of the devices
- Add more model documentation, in particular what all the output maps are
- DynamicGlobal kernel data element
    - as a substitute for Local where needed
- Add more documentation on how to change devices
