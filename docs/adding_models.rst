.. _dynamic_modules:

#############
Adding models
#############
After installing MDT you will have a folder in your home drive named ``.mdt``.
This folder contains diffusion MRI models and other functionality that you can extend without needing to reinstall or recompile MDT.

The ``.mdt`` folder contains, for every version of MDT that existed on your machine, a directory containing the configuration files and a
folder with the dynamically loadable modules.
A typical layout of the ``.mdt`` directory is:

* .mdt/
    * <version>/
        * mdt.default.conf
        * mdt.conf
        * components/


The configuration files are discussed in :ref:`configuration`, the components folder is discussed in this chapter.

The components folder consists of two sub-folders, *standard* and *user*, with an identical folder structure for the contained modules:

* components/
    * standard
        * compartment_models
        * composite_models
        * ...
    * user
        * compartment_models
        * composite_models
        * ...


By editing the contents of these folders, the user can add, extend and/or remove functionality in MDT.
The folder named *standard* contains modules that come pre-supplied with MDT.
These modules can change from version to version and any change you make in in this folder will be lost after an update.
To make persistent changes you can add your modules to the *user* folder.
The content of this folder is automatically copied to a new version.

The rest of this chapter explains the various components in more detail.

.. include:: _dynamic_modules/parameters.rst
.. include:: _dynamic_modules/compartment_models.rst
.. include:: _dynamic_modules/composite_models.rst
.. include:: _dynamic_modules/cascade_models.rst
.. include:: _dynamic_modules/library_functions.rst
.. include:: _dynamic_modules/noise_std_estimators.rst
.. include:: _dynamic_modules/batch_profiles.rst
