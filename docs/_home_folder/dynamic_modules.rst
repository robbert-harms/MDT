.. _dynamic_modules:

Dynamic modules
===============
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


.. _dynamic_modules_parameters:

Parameters
----------
Parameters form the building blocks of the compartment models. When constructing a compartment model, you have to define one or more
parameters. The semantics of these parameters determine how the data is provided to the model. For example, protocol parameters guarantee that the
corresponding data is loaded from the protocol file. Free parameters on the other hand are commonly the parameters being optimized.

The exact semantics of each parameter is determined by the class from which it inherits.
For example, compare these two parameters::

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
lower- and upper- bound, sampling prior, parameter transformation function and more. During optimization, parameters of this type can be fixed
to a specific value, which means that they are no longer optimized but that their values (per voxel) are provided by a static map.
When fixed, these parameters are still classified as free parameters to distinguish them from the other parameter types.

A free parameter is identified by having the super class :py:class:`~mdt.models.parameters.FreeParameterConfig` and
are commonly placed in the Python module named ``free.py``.

.. _protocol_parameters:

Protocol parameters
^^^^^^^^^^^^^^^^^^^
These parameters are meant to be fulfilled by the values in the Protocol (see :ref:`concepts_protocol` in Concepts). During model optimization
MDT checks for any protocol parameters and tries to match the parameter names in the model with the column names in the Protocol.
This is an important step since it allows the user to add their own column definitions to the protocol file.
If during name resolution for some protocol parameters no match can be found, MDT will issue a warning that the protocol is insufficient for the given model.

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
The compartment models form the components from which the multi-compartment models are build. They consists, in basis,
of two parts, a list of parameters (see :ref:`dynamic_modules_parameters`) and the model code in OpenCL C (the OpenCL dialect of C99).
At runtime, MDT loads the C code of the compartment model and combines it with the other compartments to form the multi-compartment model (see :ref:`concepts_cl_code`).

The compartment models must be defined in a ``.py`` file where the **filename matches** the **class name** and it only allows for **one** compartment **per file**.
For example, the following example compartment model is named ``Stick`` and must therefore be contained in a file named ``Stick.py``::

    class Stick(CompartmentConfig):

        parameter_list = ('g', 'b', 'd', 'theta', 'phi')
        cl_code = '''
            return exp(-b * d * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                              sin(phi) * sin(theta), cos(theta), 0.0)), 2));
        '''


This ``Stick`` example contains all the basic definitions required for a compartment model, a parameter list and CL code.
The elements of the parameter list can either be string, referencing one of the parameters defined in the dynamically loadable parameters (like shown here),
or it can directly be an instance of a parameter. For example, this is also a valid parameter list::

    class special_param(FreeParameterConfig):
        ...

    class MyModel(CompartmentConfig):

        parameter_list = ('g', 'b', special_param())


here the parameters ``g`` and ``b`` are loaded from the dynamically loadable parameters while the ``special_param`` is given as a parameter instance.

Splitting the CL and Python file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The CL code for a compartment model can either be given in the definition of the compartment, like shown above, or it can be provided in
a separate ``.cl`` file with the same name as the compartment.
An advantage of using an external ``.cl`` file is that you can include additional subroutines in your model definition.
The following is an example of splitting the CL code from the compartment model definition:

``Stick.py``::

    class Stick(CompartmentConfig):

        parameter_list = ('g', 'b', 'd', 'theta', 'phi')

``Stick.cl``:

.. code-block:: c

    mot_float_type cmStick(
        const mot_float_type4 g,
        const mot_float_type b,
        const mot_float_type d,
        const mot_float_type theta,
        const mot_float_type phi){

        return exp(-b * d * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                                  sin(phi) * sin(theta), cos(theta), 0.0)), 2));
    }

Note the absence of the attribute ``cl_code`` in the ``Stick.py`` file and note the naming scheme where the two filenames and the model name are exactly the same.
Also note that with this setup you will need to provide the function signature yourself. The syntax of this signature is as follows:

.. code-block:: c

    mot_float_type cm<YourModelName>(
        <type_modifiers> <param_name>,
        ...
    )

Where ``<YourModelName>`` ideally matches the name of your compartment model and the type modifier in ``<type_modifier>`` should match that of your parameter definition.
MDT commonly uses the ``mot_float_type`` which is type defined to either float or double (see :ref:`concepts_cl_code`) depending on if you use double precision or not.
The model name does not necessarily needs to match that of the filenames, but it should be unique to avoid naming conflicts during compilation.


.. _dynamic_modules_composite_models:

Composite models
----------------
The composite models, or, multi-compartment models are the models that MDT actually optimizes.
Just as the compartments are built using parameters as a building block, the composite models are built using compartments as building blocks.
Since the compartments already contain the CL code, no further model coding is necessary in the multi-compartment models.
When asked to optimize (or sample) a model, MDT combines the CL code of the compartments into one objective function and uses the
parameters of the compartments to load the correct data.

In contrast to the compartment models which must be placed in their own file, the composite models can be placed in any ``.py`` file within the ``composite_models`` directory.
The following is an minimum example of a composite (multi-compartment) model in MDT::

    class BallStickStick(DMRICompositeModelConfig):

        model_expression = '''
            S0 * ( (Weight(w_ball) * Ball) +
                   (Weight(w_stick0) * Stick(Stick0)) +
                   (Weight(w_stick1) * Stick(Stick1)) )
        '''

The model expression is a string that expresses the model in a MDT model specific mini-language.
This language, which only accepts the operators ``*``, ``/``, ``+`` and ``-`` can be used to combine your compartments in any way possible (within the grammar of the mini-language).
MDT parses this string, loads the compartments from the compartment models and uses the CL code of these compartments to create the CL objective function for your model.

The example above combines the compartments (``Ball`` and ``Stick``) as a weighted summation using the special compartment ``Weight`` for the compartment weighting
(these weights are sometimes called volume fractions).
The example also shows compartment renaming.
Since it is possible to use a compartment multiple times, it is necessary to rename the double compartments to ensure that all the compartments have a unique name.
This renaming can be done by specifying the renamed model name in parenthesis after the compartment model name.
For example ``Stick(Stick0)`` refers to a ``Stick`` compartment that has been renamed to ``Stick0``. This new name is then used to refer to that specific compartment in the
rest of the composite model attributes.

todo:
* parameter dependencies
* weights have auto summation
* additional output maps

.. _dynamic_modules_composite_models_protocol_options:

Protocol options
^^^^^^^^^^^^^^^^


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
