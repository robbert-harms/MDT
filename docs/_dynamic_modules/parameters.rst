.. _dynamic_modules_parameters:

**********
Parameters
**********
Parameters form the building blocks of the compartment models.
They define how data is provided to the model and form a bridge between the model and the :ref:`concepts_input_data_models`.

The type of a parameters determines how the model uses that parameter.
For example, compare these two parameters::

    class theta(FreeParameterTemplate):
        ...

    class b(ProtocolParameterTemplate):
        ...


In this example the ``theta`` parameter is defined as a *free parameter* while the ``b`` parameter is defined as a *protocol parameter*.
The type matters and defines how MDT handles the parameter.
There are only two types available:

* :py:class:`~mdt.models.parameters.FreeParameterTemplate`, for :ref:`free_parameters`
* :py:class:`~mdt.models.parameters.ProtocolParameterTemplate`, for :ref:`protocol_parameters`

See the sections below for more details on each type.


.. _free_parameters:

Free parameters
===============
These parameters are normally supposed to be optimized by the optimization routines.
They contain some meta-information such as a lower- and upper- bound, sampling prior, parameter transformation function and more.
During optimization, parameters of this type can be fixed to a specific value, which means that they are no longer optimized
but that their values (per voxel) are provided by a scalar or a map.
When fixed, these parameters are still classified as free parameters (you can consider them as fixed free parameters).

To fix these parameters you can either define so in a composite model, a cascade model or using the Python API before model optimization

.. code-block:: python

    mdt.fit_model('CHARMED_r1',
                  ...,
                  initialization_data=dict(
                      fixes={},
                      inits={}
                  ))


A free parameter is identified by having the super class :py:class:`~mdt.component_templates.parameters.FreeParameterTemplate`.

Hereunder we list some details that are important when adding a new free parameter to MDT.

Parameter transformations
-------------------------
Panagiotaki (2012) and Harms (2017) describe the use of parameter transformations to limit the range of each parameter
to biophysical meaningful values and to scale the parameters to a range better suited for optimization.
They work by injecting a parameter transformation before model evaluation that limits the parameters between bounds.
See (Harms 2017) for more details on which transformations are used in MDT.
You can define the transformation function used by setting the ``parameter_transform`` attribute.
For an overview of the available parameter transformations, see :mod:`~mdt.model_building.parameter_functions.transformations` in MOT.

Sampling
--------
For sampling one needs to define per parameter a prior and a proposal function.
These can easily be added in MDT using the attributes ``sampling_proposal`` and ``sampling_prior``.
Additionally, one can define a sampling statistic function ``sampling_statistics`` which is ran after sampling and returns statistics on the observed samples.


.. _protocol_parameters:

Protocol parameters
===================
These parameters are meant to be fulfilled by the values in the Protocol file (see :ref:`concepts_protocol`) or in the extra protocol maps.
During model optimization, MDT checks the model for protocol parameters and tries to match the names of the protocol parameters with available protocol data.
Protocol data can be submitted using either the Protocol file, or using voxel-based protocol maps.

By matching parameter names with input values, the user can add protocol values dynamically by ensuring a common name between the protocol parameter and the provided values.

A protocol parameter is identified by having the super class :py:class:`~mdt.component_templates.parameters.ProtocolParameterTemplate`.
