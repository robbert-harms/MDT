.. _configuration:

Configuration
=============
The default configuration is loaded inside MDT and can be viewed in your home folder in ``.mdt/<version>/mdt.default.conf``.
That file is there for reference and changing it will not affect the functioning of MDT. To override the default configuration,
you can copy this file to ``.mdt/<version>/mdt.conf`` and set (only) the options you wish to override. These files
change the configuration of MDT at application startup, to change these settings at runtime you can use runtime configurations (see below).


General config
--------------
The default configuration file is stored in ``.mdt/<version>/mdt.default.conf`` and is only there for reference. The format of this file
is the **YAML** format. To change part of the configuration you can make a copy of this file to ``.mdt/<version>/mdt.conf`` and set the options you want to change.

For example, suppose you want to disable the automatic zipping of the nifti's of the optimization results.
You can create an *empty* ``mdt.conf`` file and add to this the lines:

.. code-block:: yaml

    output_format:
        optimization:
            gzip: False
        sampling:
            gzip: False


This leaves the rest of the configuration file to its defaults and only changes the output format specification. If you also wish to change the default value of
``tmp_results_dir`` you can add it (in any order) to your ``mdt.conf`` file:

.. code-block:: yaml

    output_format:
        optimization:
            gzip: False
        sampling:
            gzip: False

    tmp_results_dir: /tmp


This file is automatically copied to the directory of a new version and only breaks if the layout of the configuration file changes.

GUI specific config
-------------------
Next to the more general ``.mdt/<version>/mdt.conf`` which is always applied, you can also have a GUI specific configuration file as ``.mdt/<version>/mdt.gui.conf``.
This file is loaded after the regular ``mdt.conf`` configuration and can override all prior values. Its purpose is to contain a few GUI specific configuration values
as well as to be able to set a specific configuration for the GUI only. The structure of the file is the same as the other configuration files.


Runtime configuration
---------------------
It is possible to change the configuration in Python code using configuration contexts. Contexts are a Python programming pattern
that allows you to run code prior and after another piece of code. MDT uses this pattern to execute a piece of code with a modified configuration, while
making sure that the configuration is reset back to its previous state after the code has been executed. See the :py:mod:`mdt.configuration` module for more details.
