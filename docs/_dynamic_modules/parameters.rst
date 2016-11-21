.. _dynamic_modules_parameters:

**********
Parameters
**********
Parameters form the building blocks of the compartment models.
They define how data is provided to the model and form a bridge between the model and the :ref:`concepts_protocol` in the problem data.

The type of a parameters determines how the model uses that parameter.
For example, compare these two parameters::

    class theta(FreeParameterConfig):
        ...

    class b(ProtocolParameterConfig):
        ...


In this example the ``theta`` parameter is defined as a *free parameter* while the ``b`` parameter is defined as a *protocol parameter*.
The type matters and defines how MDT handles the parameter when added to a compartment model.
There are four types available:

* :py:class:`~mdt.models.parameters.FreeParameterConfig`, for :ref:`free_parameters`
* :py:class:`~mdt.models.parameters.ProtocolParameterConfig`, for :ref:`protocol_parameters`
* :py:class:`~mdt.models.parameters.ModelDataParameterConfig`, for :ref:`model_data_parameters`
* :py:class:`~mdt.models.parameters.StaticMapParameterConfig`, for :ref:`static_map_parameters`

See the sections below for more details on each type.


.. _free_parameters:

Free parameters
===============
These parameters are supposed to be optimized by the optimization routines. They contain some meta-information such as a
lower- and upper- bound, sampling prior, parameter transformation function and more. During optimization, parameters of this type can be fixed
to a specific value, which means that they are no longer optimized but that their values (per voxel) are provided by a static map.
When fixed, these parameters are still classified as free parameters to distinguish them from the other parameter types.

A free parameter is identified by having the super class :py:class:`~mdt.models.parameters.FreeParameterConfig` and
are commonly placed in the Python module named ``free.py``.

Hereunder we list some details that are important when adding a new Free Parameter to MDT.

Parameter transformations
-------------------------
Panagiotaki (2012) and Harms (2017) describe the use of parameter transformations to limit the range of each parameter
to biophysical meaningful values and to scale the parameters to a range better suited for optimization.
They work by injecting a parameter transformation before model evaluation that limits the parameters between bounds.
See (Harms 2017) for more details on which transformations used.
You can define the transformation function used by setting the ``parameter_transform`` attribute.
For an overview of the available parameter transformations, see :mod:`~mot.model_building.parameter_functions.transformations` in MOT.

Sampling
--------
For sampling one needs to define per parameter a prior and a proposal function.
These can easily be added in MDT using the attributes ``sampling_proposal`` and ``sampling_prior``.
Additionally, one can define a sampling statistic function ``sampling_statistics``.
This function is run after sampling and should return basic statistics on the observed samples.


.. _protocol_parameters:

Protocol parameters
===================
These parameters are meant to be fulfilled by the values in the Protocol (see :ref:`concepts_protocol` in Concepts).
During model optimization MDT checks for present protocol parameters and tries to match the names of the parameters with the names of the columns in the Protocol.
This setup allows the user to add their own column definitions to the protocol file, only by ensuring a common name between the protocol parameter and the protocol column name.

The values in the protocol are assumed to be constant over voxels and dynamic over volumes.
That is, the values in the protocol file have, for each column, one value per volume.
That value is then used for every voxel in that volume. To have static values that are dynamic per volume and per voxel, use :ref:`static_map_parameters`.

A protocol parameter is identified by having the super class :py:class:`~mdt.models.parameters.ProtocolParameterConfig` and
are commonly placed in the Python module named ``protocol.py``.

.. _static_map_parameters:

Static map parameters
=====================
The static map parameters are meant to carry additional observational data about a problem.
When defined, MDT tries to load the appropriate data from the ``static_maps`` the problem data (see :ref:`concepts_problem_data_models`).

The values in the static maps are meant for values per voxel and optionally also per volume.
They can hold, for example, b0 inhomogeneity maps or flip angle maps that have a specific value per voxel and (optionally) per volume.

A static map parameter is identified by having the super class :py:class:`~mdt.models.parameters.StaticMapParameterConfig` and
are commonly placed in the Python module named ``static_maps.py``.


.. _model_data_parameters:

Model data parameters
=====================
These parameters are meant for model specific data that the model needs to function correctly. You can inline these variables in
the compartment model CL code (which is faster), but than the end-users can not easily change these values. By adding them as
model data parameters, end-users can change the specifics of the model by changing the data in the model data parameters.
They are not commonly used and are of a more technical kind than the other parameters.

A model data parameter is identified by having the super class :py:class:`~mdt.models.parameters.ModelDataParameterConfig` and
are commonly placed in the Python module named ``model_data.py``.

