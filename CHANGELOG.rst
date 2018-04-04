*********
Changelog
*********

v0.10.9 (2018-02-22)
====================

Added
-----
- Adds covariance singularity boolean matrix to the output results.

Fixed
-----
- Fixed small bug in the mdt maps visualizer. Refactored the batch fitting function to use the batch apply function.


v0.10.8 (2018-02-16)
====================

Changed
-------
- Updated the map view config syntax for the voxel highlights (now called annotations).
- Updates following MOT in DKI measures.
- Changed the config layout of the maps visualizer with regards to the colorbar settings.


v0.10.7 (2018-02-14)
====================

Changed
-------
- Changed the parameter proposal and transform function of the PHI parameter.

Fixed
-----
- Fixes issue #4, the MDT gui crashed on startup with Qt version 5.9.1.


v0.10.6 (2018-01-30)
====================

Added
-----
- Adds colormap order in the GUI when a map is interpreted as colormap.
- Adds relaxometry models.
- Adds sampling output selection to the sampler.
- Adds another post-processing switch to the sampling post-processing.
- Adds nibabel and numpy array decoration to store path info alongside the niftis when loaded with mdt.load_nifti().
- Adds Hessian and covariance computation as post-processing to the models.

Changed
-------
- Updates to the batch profiles.
- Updates to CHARMED boundary conditions.

Other
-----
- Removed the sampling statistics calculation from the post-processing, it did not work out theoretically.
- Adds an utility function for computing the correlations from the covariances.
- Small update to the scientific scrollers in the gui. Interchanged the position of max and min in the gui.
- Renamed evaluation_model to likelihood_function in the composite models. This covers the usage better.


v0.10.5 (2017-09-22)
====================

Added
-----
- Adds support for multiple output files in the mdt-math-img CLI function.
- Adds post sampling log messages
- Adds caching to deferred loading collections.

Changed
-------
- Changed the signature of write_nifti and moved the header argument to the optional keyword arguments.
- Updates to the documentation of the configuration.
- Small improvements in the post-sampling processing.
- the function ``write_nifti`` now creates the directories if they do not exist yet.

Fixed
-----
- Fixed non working documentation build on read the docs. Removed the ``sphinxarg.ext`` since it is not supported yet on read the docs.

Other
-----
- Small path updates to the batch profiles.
- MDT now also saves the log likelihood and log priors after sampling.
- Made the sampler sample from the complete log likelihood. This allows storing the likelihood and prior values and use them later for maximum posterior and maximum likelihood calculations.
- Simplified model compartment expressions due to improvements in MOT.


v0.10.4 (2017-09-06)
====================

Changed
-------
- Changes the default sampling settings of the phi parameter. Since it is supposed to wrap around 2*pi, we can not use the circular gaussian approximation if we are constraining it between 0 and pi, instead we use a simple gaussian proposal and a truncated gaussian sampling estimate.
- Updates to the processing strategies. Adds an interface for MRIModels to work with the processing strategies.

Other
-----
- Following the changes in MOT, we can now let a compartment model and a library function evaluate itself given some input data.


v0.10.3 (2017-08-29)
====================

Added
-----
- Adds some of the new config switches in the maps visualizer to the graphical interface.
- Adds the possibility of interpreting vector maps as RGB maps. Useful for displaying Tensor FA orientation maps.
- Added overridden method to the problem data.
- Adds support for fitting when the protocol is empty.
- Added parameter name logging to MDT instead of in MOT.

Changed
-------
- Updated the processing strategy with a better mask file placement (technical thing).
- Updates to the sampling post-processing.
- Updates to the documentation.
- Updated the InputDataMRI interface to contain a few more properties.
- Updated the changelog generation slightly.
- Updated the ExpT1DecIR model, adds a cascade. Updated the way cascades are updated as such that it allows for multiple copies of the same model in a cascade.
- Updates to the GUI.
- Updates the parser to the latest version of Grako.

Fixed
-----
- Fixed naming issues when loading new maps in the map viewer.
- Fixes the image squeezing in the viewer when adding a colorbar.
- Fixed the issue with the get_free_param_names removal.

Other
-----
- Version bump.
- Small refactoring in the processing strategy.
- Renamed the S0-TIGre model to S0_TI_GRE.
- Reverted some changes on the S0-T1-GRE model.
- Renamed InputDataMRI to MRIInputData and InputDataDMRI to SimpleMRIInputData.
- Renamed 'problem_data' to 'input_data', 'DMRIProblemData' to 'InputDataDMRI' and all other possible renamings. This also deprecates the function  since it has been renamed to .
- Following changes in MOT.


v0.10.2 (2017-08-23)
====================

Added
-----
- Adds chunk indices look-a-head in the processing strategies. This allows the Processor to start pre-loading the next batch.


v0.10.1 (2017-08-22)
====================

Changed
-------
- Updates to the GUI.
- Updates to the maps visualizer.


v0.10.0 (2017-08-17)
====================

Added
-----
- Adds automatic changelog generation from the git log.
- Adds multivariate statistic to sampling output. Changes the KurtosisExtension to a KurtosisTensor single model.
- Adds catch for special case.
- Adds Tensor reorientation as a post processing. This reorients theta, phi and psi to match the sorted eigen vectors / eigen values.
- Adds compartment model sorting based on weights as a post-processing to composite models. Adds automatic sorting to Ball&Sticks and CHARMED models.
- Adds small boundary conditions to the Kurtosis model.
- Adds clickable point information to the map visualization GUI.
- Adds name collision resolution in the visualization GUI after dragging in images with the same name.
- Adds a library function for the kurtosis matrix multiplication.
- Added component construction to the __new__ of a component template. This allows the template to construct itself at object initialization.

Changed
-------
- Changes the way the logging is condensed during optimization.
- Updates to the GUI.
- Updates to the documentation. Also, the compartment models now no longer need their own files, they can be defined in any file in the compartment_models directory.
- Updates to the documentation, renamed the Kurtosis compartment to KurtosisExtension and made it require the Tensor in the Composite model.
- Updates to the documentation. Updates to the Kurtosis model. Sets boundary conditions correct and adds post-processing.
- Updates to the documentation style.

Fixed
-----
- Fixed bug in matplotlib renderer with the highlight voxel.
- Fixed the small GUI bug with the random maps naming.

Other
-----
- Removed calculated example files.
- Removed redundant logging.
- Small renaming updates.
- Adds some linear algebra methods to the utilities, Changed the way the psi component of the Tensor is used.
- More work on the post-sampling statistics.
- Removed redundant model.
- Moved more relaxometry compartments to the single python file. Slightly increased the number of voxels in sampling.
- Update to the cartesian to spherical function.
- First work on map sorting.
- Small bugfix in the MRI constants.
- Small function reshuffling, updates to comments.
- Small fix with the InitializationData in the fit model.
- Small bugfix to the GUI.
- Completely adds the Kurtosis model. Adds some small library functions as well for the Tensor and Kurtosis computations.


v0.9.40 (2017-07-27)
====================

Added
-----
- Adds ActiveAx cascade.

Other
-----
- Small release to add ActiveAx cascade model.
- Small update to docs.


v0.9.39 (2017-07-26)
====================

Changed
-------
- Updates to the documentation

Other
-----
- Small fix allowing b-value to be stored in protocol alongside Delta, delta and G.
- Removed the functionality of having the CL code in a separate file for the compartment models and the library models. Now everything is in the Python model definition.


v0.9.38 (2017-07-25)
====================

Added
-----
- Adds Kurtosis model.
- Adds the extra-axonal time dependent CHARMED from (De Santis 2016). Still needs to be tested though.
- Adds TimeDependentZeppelin for use in the extra-axonal time dependent CHARMED model. Also, the dependency_list in the compartments now also accepts other compartments as strings. Finally, the compartments now no longer need the prefix "cm" in their CL callable function"
- Adds the ActiveAx model.
- Adds the ActiveAx model, slight update to what the Neumann cylindrical function calculates.

Changed
-------
- Updates in the new version:
  - Small update in the model fit GUI, separated the models from the
    cascades to make it more clear what these mean
  - Adds three new models:
      - ActiveAx
      - Time Dependent ActiveAx (see De Santis 2016)
      - Kurtosis
  - Simplified the processing strategies to make it more robust
  - The visualization GUI can now load images from multiple folders
  - The visualization GUI now also supports dragging nifti files into
    the viewer for loading and viewing.
- Updates to some of the relaxometry models, fixed the simulations to the latest MOT version.

Fixed
-----
- Fixed list/dict bug in viewer.
- Fixed the simulations module to work with the latest MOT version. Updates to some of the relaxometry models.

Other
-----
- Small documentation update.
- Update to Kurtosis.
- Merge branch 'master' of github.com:cbclab/MDT.
- Merged local copy, fixed small issue in the dragging of files in the visualization GUI.
- Some initial work on the AxCaliber model. We are not there yet.
- More simplifications to the models, adds reload function in the module loaders (for reloading the cache), add TemplateModifier that can rewrite the source code of a template.
- Merge branch 'master' of github.com:cbclab/MDT.
- In the model fit GUI, separated the models from the cascades to make it more clear what the cascades do.
- In the model fit GUI, separated the models from the cascades to make it more clear what the cascades do.
- Renamed the Silvia 2016 time dependent model from CHARMED to ActiveAx.
- Made ActiveAx diffusivity dependency more clear.
- Removed the GDRCylindersFixedRadii compartment model, it was not used anywhere. Simplified the NODDI tortuosity parameter dependency.
- Update to doc about the parameter renaming.
- The parameter definitions in the compartment model now support nicknaming to enable swapping a parameter without having to rename that parameter in the model equation or other code.
- Renamed the component_configs to component templates and moved some base classes to other folders. Also, all components constructed from templates now carry a back reference to that template as a class attribute.
- Small updates to the processing strategies.
- Prepared the processing strategies for possible multithreading.
- Small comment update in the processing strategy.
- Refactored the processing strategies such that paralellization may be possible in the future.


