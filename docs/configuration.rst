.. _configuration:

#############
Configuration
#############
The default configuration can be viewed in your home folder in ``.mdt/<version>/mdt.default.conf``.
This file is merely there for reference and is not read by MDT (rather, those defaults are loaded within MDT).
To override the default configuration you can copy this file to ``.mdt/<version>/mdt.conf`` and set (only) the options you wish to override.
These configuration files change the configuration of MDT at application startup, to apply a new configuration file you will need to restart MDT.

.. contents:: Table of Contents
   :local:
   :backlinks: none


**************
General config
**************
The default configuration file is stored in ``.mdt/<version>/mdt.default.conf`` and is only there for reference.
The configuration is in **YAML** format.
To change part of the configuration you can make a copy of this file to ``.mdt/<version>/mdt.conf`` and set the specific options you want to change.

For example, suppose you want to disable the automatic zipping of the nifti files after optimization.
You can create an *empty* ``mdt.conf`` file and add to this the lines:

.. code-block:: yaml

    output_format:
        optimization:
            gzip: False
        sampling:
            gzip: False


This leaves the rest of the configuration file at the defaults.
If you also wish to change the default value of ``tmp_results_dir`` you can add it (in any order) to your ``mdt.conf`` file:

.. code-block:: yaml

    output_format:
        ...

    tmp_results_dir: /tmp


The ``mdt.conf`` file is automatically copied to the directory of a new version.
The configuration options are stable in general and may only break over major versions.


*******************
GUI specific config
*******************
Next to the more general ``.mdt/<version>/mdt.conf`` which is applied in general, you can also have a GUI specific configuration file, ``.mdt/<version>/mdt.gui.conf``.
When you start any of the MDT graphical interfaces, this file is loaded after the regular ``mdt.conf`` configuration and hence takes priority over the other settings.
Its purpose is to contain a few GUI specific configuration values as well as to be able to set a specific configuration for the GUI only.
The structure of the file is the same as that of the other configuration files.


*********************
Runtime configuration
*********************
It is also possible to change the MDT configuration during code execution using configuration contexts.
Contexts are a Python programming pattern that allows you to run code before and after another piece of code.
MDT uses this pattern to allow for temporarily changing the MDT configuration for the duration of the context.
An advantage of this pattern is that is makes sure that the configuration is reset back to its previous state after the code has been executed.
An example of using the configuration contexts:

.. code-block:: python

    with mdt.config_context(ConfigAction()):
        mdt.fit_model(...)

See the :py:mod:`mdt.configuration` module for more details on how to use this functionality.
