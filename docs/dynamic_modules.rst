.. _dynamic_modules:

***************
Dynamic modules
***************

.. contents:: Table of Contents
   :local:
   :backlinks: none


At application startup MDT automatically loads certain modules from a folder on your home drive.
These modules are Python files containing functionality that the user can extend without the need to reinstall or recompile MDT.

After installing MDT you will have a folder in your home drive named ``.mdt``. This folder contains, for every version of MDT that existed on your machine,
a directory containing the configuration files and a folder with the dynamically loadable modules. A typical layout of the ``.mdt`` directory is:

* .mdt
    * <version>
        * ``mdt.default.conf``
        * ``mdt.conf``
        * components
            * standard
                * batch_profiles
                * cascade_models
                * ...
            * user
                * batch_profiles
                * cascade_models
                * ...


The configuration files are discussed in :ref:`configuration`.
The components folder is split into two parts, *standard* and *user* with an identical folder structure. By editing the
contents of these folders, the user can add, extend and/or remove functionality in MDT. The folder named *standard* contains modules
that come pre-supplied with MDT. These modules can change from version to version and any change you make in in this folder will be lost
after an upgrade to a new version. If you want to persist your changes from version to version you can add your modules to the *user* folder.
The content of this folder is automatically copied to a new version.
For example, to add a custom Batch Profile, copy one of the existing batch profiles from the standard folder to the user folder and adapt it to your liking.
At application startup it is then automatically picked up by MDT.
In the case of naming conflicts the modules in the user folder take priority over the modules in the standard folder.

The rest of this chapter explains the various components in more detail.

.. include:: _dynamic_modules/parameters.rst
.. include:: _dynamic_modules/compartment_models.rst
.. include:: _dynamic_modules/composite_models.rst
.. include:: _dynamic_modules/cascade_models.rst
.. include:: _dynamic_modules/library_functions.rst
.. include:: _dynamic_modules/noise_std_estimators.rst
.. include:: _dynamic_modules/processing_strategies.rst
.. include:: _dynamic_modules/batch_profiles.rst
