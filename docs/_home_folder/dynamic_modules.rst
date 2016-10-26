.. _dynamic_modules:

Dynamic modules
===============
MDT automatically loads certain modules at application startup from a folder on your home drive.
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


The components folder is split into two parts, *standard* and *user*, each containing the same folder structure. By editing the
contents of these folders, the user can add, extend and remove functionality in MDT. The folder named *standard* contains modules
that come pre-supplied with MDT. These modules can change from version to version and any change you make in in this folder will be lost
after a upgrade to a new version. Modules that you want to persist from version to version can be added to the *user* folder. The content of this folder
is automatically copied to a new version.

For example, to add a custom Batch Profile, copy one of the existing batch profiles from the standard folder to the user folder and adapt it to your liking.
Afterwards it is automatically picked up by MDT.

The modules in the user folder take priority over the modules in the standard folder in the case of equal names.


.. _dynamic_modules_parameters:

Parameters
----------
Every compartment model consists of one or more parameters. The exact semantics of each parameter is determined by the
class from which it inherits. For example, compare these two parameters::

    class theta(FreeParameterConfig):
        ...

    class b(ProtocolParameterConfig):
        ...


In this example the ``theta`` parameter is defined as a *free parameter* while the ``b`` parameter is defined as a *protocol parameter*.
The type matters and defines how MDT handles the parameter when added to a compartment model. There are four types available:

* :py:class:`~mdt.models.parameters.FreeParameterConfig`, for :ref:`free_parameters`
* :py:class:`~mdt.models.parameters.ProtocolParameterConfig`, for :ref:`protocol_parameters`
* :py:class:`~mdt.models.parameters.ModelDataParameterConfig`, for :ref:`model_data_parameters`
* :py:class:`~mdt.models.parameters.StaticMapParameterConfig`, for :ref:`static_map_parameters`

See the sections below for more details on each type.

.. _free_parameters:

Free parameters
^^^^^^^^^^^^^^^
These parameters are supposed to be optimized by the optimization routines. They contain some meta-information such as a
lower- and upper- bound, sampling prior, parameter transformation function and so forth. During optimization, parameters of this type may be fixed
to a specific value, which means that they are no longer optimized but that their values (per voxel) are provided by the fixed value.
When fixed they are still classified as free parameters to distinguish them from the other parameter types.

A free parameter is identified by having the super class :py:class:`~mdt.models.parameters.FreeParameterConfig` and
are commonly placed in the Python module named ``free.py``.

.. _protocol_parameters:

Protocol parameters
^^^^^^^^^^^^^^^^^^^
These parameters are meant to be fulfilled by the values in the Protocol (see :ref:`concepts_protocol` in Concepts). During model optimization
MDT checks the model for protocol parameters and tries to match the parameter names with the column names in the Protocol.
If for some protocol parameters no match can be found, MDT issues a warning that the protocol is insufficient for the given model.

The values in the protocol are assumed constant over voxels and dynamic over volumes. That is, the values in the protocol file have, for each column, one value per volume.
That value is then used for every voxel in that volume. To have static values that are dynamic per volume and per voxel, use :ref:`static_map_parameters`.

A protocol parameter is identified by having the super class :py:class:`~mdt.models.parameters.ProtocolParameterConfig` and
are commonly placed in the Python module named ``protocol.py``.


.. _static_map_parameters:

Static map parameters
^^^^^^^^^^^^^^^^^^^^^
The static map parameters are meant to carry additional observational data about a problem. When defined, MDT tries to load
the appropriate data from either the problem data (see :ref:`concepts_problem_data_models`) or from the default value in the parameter definition.

The values in the static maps are meant for values per voxel and optionally also per volume. They can hold, for example, b0 inhomogeneity maps or flip angle maps that
have a specific value per voxel and (optionally) per volume.

A static map parameter is identified by having the super class :py:class:`~mdt.models.parameters.StaticMapParameterConfig` and
are commonly placed in the Python module named ``static_maps.py``.


.. _model_data_parameters:

Model data parameters
^^^^^^^^^^^^^^^^^^^^^
These parameters are meant for model specific data that the model needs to function correctly. You can inline these variables in
the compartment model CL code (which is faster), but than the end-users can not easily change these values. By adding them as
model data parameters, end-users can change the specifics of the model by changing the data in the model data parameters.
They are not commonly used and are of a more technical kind than the other parameters.

A model data parameter is identified by having the super class :py:class:`~mdt.models.parameters.ModelDataParameterConfig` and
are commonly placed in the Python module named ``model_data.py``.



Compartment model
-----------------
The compartment models form the components from which the multi-compartment models are build. They consists, in basis, of
two parts, a list of parameters (see :ref:`dynamic_modules_parameters`) and the model code in OpenCL C (the OpenCL dialect of C99).
At runtime MDT loads the C code of the compartment and combines it with the other compartments to form the multi-compartment model.
The parameters and their configuration are used to load the correct data from the :ref:`problem data <concepts_problem_data_models>` during, for example, model optimization.

The following is an example compartment model expression, copied from the Stick compartment in MDT::

    class Stick(CompartmentConfig):

        parameter_list = ('g', 'b', 'd', 'theta', 'phi')
        cl_code = '''
            return exp(-b * d * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                              sin(phi) * sin(theta), cos(theta), 0.0)), 2));
        '''


this example contains all the basic definitions required for a compartment model.
The elements of the parameter list can either be a parameter instance or it can be a reference to one of the parameters defined in the dynamically loadable parameters.
Hence, this is also a valid parameter list::


    class my_special_param(FreeParameterConfig):
        ...

    class MyModel(CompartmentConfig):

        parameter_list = ('g', 'b', my_special_param())



here the parameters ``g`` and ``b`` are loaded from the parameters





Single models
-------------
todo


Cascade models
--------------
todo

Library functions
-----------------
todo

Noise std. estimators
---------------------
todo

Processing strategies
---------------------
todo

Batch profiles
--------------
todo
