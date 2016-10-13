.. _dynamic_modules:

Dynamic modules
===============
MDT can automatically load certain modules at application startup from a folder on your home drive.
These modules are Python files containing functionality that the user can extend without the need to reinstall or recompile MDT.

After installation you will have a folder in your home drive named ``.mdt``. This folder contains for every version of MDT that existed on your machine
a directory containing the dynamically loadable modules and the configuration files. A typical layout is of the ``.mdt`` directory is:

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


The components folder is split up into two parts, *standard* and *user* each containing the same set of folders. By editing the
contents of these folders the user can add, extend and remove functionality in MDT. The folder named *standard* contains modules
that come pre-supplied with MDT. The modules can change from version to version and any change you make in standard will be lost
in a newer version of MDT. To add modules that persist from version to version, please add your modules to the *user* folder.
The user folder is copied automatically to a new version.

For example, if you wish to add a custom Batch Profile to MDT you can copy one of the existing batch profiles from the standard folder
to the user folder and adapt it to your liking.

The modules in the user folder take priority over the modules in the standard folder in the case of equal names.



Parameters
----------
Every compartment model consists of one or more parameters. The exact semantics of each parameter is determined by the
class from which it inherits. For example, compare these two parameters::

    class theta(FreeParameterConfig):
        ...

    class b(ProtocolParameterConfig):
        ...


The ``theta`` parameter is here defined as a *free parameter* while the ``b`` parameter is defined as a protocol parameter.
The type matters and defines how MDT will handle the parameter when added to a compartment model. There are four types available:

* :py:class:`~mdt.models.parameters.FreeParameterConfig`, for :ref:`free_parameters`
* :py:class:`~mdt.models.parameters.ProtocolParameterConfig`, for :ref:`protocol_parameters`
* :py:class:`~mdt.models.parameters.ModelDataParameterConfig`, for :ref:`model_data_parameters`
* :py:class:`~mdt.models.parameters.StaticMapParameterConfig`, for :ref:`static_map_parameters`

.. _free_parameters:

Free parameters
^^^^^^^^^^^^^^^
These parameters are supposed to be optimized by the optimization routines. They contain some meta-information such as a
lower- and upper- bound, sampling prior, parameter transform and so forth. During optimization they may be fixed
to a specific value, which means they are no longer optimized by the optimization routines but their values (per voxel)
are given by the fixed value. When fixed they are still classified as free parameters to distinquish them from the other three types.

A free parameter is identified by having the super class :py:class:`~mdt.models.parameters.FreeParameterConfig` and
are commonly placed in the Python module named ``free.py``.

.. _protocol_parameters:

Protocol parameters
^^^^^^^^^^^^^^^^^^^
These parameters are meant to be fulfilled by values in the Protocol file (see :ref:`concepts_protocol` in Concepts). During model optimization
MDT checks the model for any parameters of this type and tries to match parameter names with the column names in the protocol file.
If no match can be found MDT issues a warning that the protocol is insufficient for the given model.

The values in the protocol are constant over voxels and dynamic over volumes. To have static values dynamic per volumes and per voxels use :ref:`static_map_parameters`.

A protocol parameter is identified by having the super class :py:class:`~mdt.models.parameters.ProtocolParameterConfig` and
are commonly placed in the Python module named ``protocol.py``.

.. _model_data_parameters:


.. _static_map_parameters:

Static map parameters
^^^^^^^^^^^^^^^^^^^^^
The static map parameters are meant to carry additional observational data about a problem. When defined, MDT tries to load
the appropriate data (either pre-supplied in the problem data (see :ref:`concepts_problem_data_models`) or from the default value in the parameter definition.

The values in the static maps are meant for values per voxel and optionally also per volume. They can carry for example b0 inhomogeneity maps or flip angle maps that
have a specific value per voxel and (optionally) per volume.

A static map parameter is identified by having the super class :py:class:`~mdt.models.parameters.StaticMapParameterConfig` and
are commonly placed in the Python module named ``static_maps.py``.


Model data parameters
^^^^^^^^^^^^^^^^^^^^^
These parameters are meant for model specific data that the model needs to function correctly. You can of course inline these variables in
the compartment model code for (which is faster), but this way lets the user change the specifics of the model by changing the data in the model data parameters.
They are not commonly used and are of a more technical kind than the other parameters.

A model data parameter is identified by having the super class :py:class:`~mdt.models.parameters.ModelDataParameterConfig` and
are commonly placed in the Python module named ``model_data.py``.



Compartment model
-----------------



Single models
-------------


Cascade models
--------------


Library functions
-----------------


Noise std. estimators
---------------------


Processing strategies
---------------------


Batch profiles
--------------
