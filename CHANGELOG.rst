*********
Changelog
*********


v0.10.0 (2017-08-17)
====================

Added
-----
- Adds automatic changelog generation from the git log.
- Adds multivariate statistic to sampling output.
- Adds Tensor reorientation as a post processing. This reorients theta, phi and psi to match the sorted eigen vectors / eigen values.
- Adds compartment model sorting based on weights as a post-processing to composite models.
- Adds automatic sorting to Ball&Sticks and CHARMED models.
- Adds small boundary conditions to the Kurtosis model.
- Adds clickable point information to the map visualization GUI.
- Added component construction to the __new__ of a component template. This allows the template to construct itself at object initialization.
- Adds condensed logging to the model fitting and sampling.

Changed
-------
- Updates to the GUI.
- Update to the cartesian to spherical function.
- Changed the loading of the compartment models, they now no longer need their own files and can be defined in any file in the compartment_models directory.
- Updates to the documentation.
- Update to the Tensor compartment model, faster way of calculating the psi rotation

Fixed
-----
- Fixed bug in matplotlib renderer with the highlight voxel.
- Fixed the small GUI bug with the random maps naming.


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
- Updates to the documentation.

Other
-----
- Small fix allowing b-value to be stored in protocol alongside Delta, delta and G.
- Removed the functionality of having the CL code in a separate file for the compartment models and the library models. Now everything is in the Python model definition.


v0.9.38 (2017-07-25)
====================

Added
-----
- Adds first draft of the Kurtosis model.
- Adds the extra-axonal time dependent CHARMED from (De Santis 2016). Still needs to be tested though.
- Adds TimeDependentZeppelin for use in the extra-axonal time dependent CHARMED model. Also, the dependency_list in the compartments now also accepts other compartments as strings. Finally, the compartments now no longer need the prefix "cm" in their CL callable function"
- Adds the ActiveAx model.

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
- Update to Kurtosis.
- Update to doc about the parameter renaming.
- Updates to some of the relaxometry models, fixed the simulations to the latest MOT version.

Fixed
-----
- Fixed list/dict bug in viewer.
- Fixed the simulations module to work with the latest MOT version. Updates to some of the relaxometry models.

Other
-----
- Small documentation update.
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
- The parameter definitions in the compartment model now support nicknaming to enable swapping a parameter without having to rename that parameter in the model equation or other code.
- Renamed the component_configs to component templates and moved some base classes to other folders. Also, all components constructed from templates now carry a back reference to that template as a class attribute.
- Small updates to the processing strategies.
- Prepared the processing strategies for possible multithreading.
- Small comment update in the processing strategy.
- Refactored the processing strategies such that paralellization may be possible in the future.


