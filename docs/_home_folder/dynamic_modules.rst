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

Hereunder we list some details that are important when adding a new parameter to MDT.

Parameter transformations
$$$$$$$$$$$$$$$$$$$$$$$$$
Panagiotaki (2012) and Harms (2017) describe the use of parameter transformations to limit the range of each parameter
to biophysical meaningful values and to scale the parameters to a range better suited for optimization.
They work by injecting a parameter transformation before model evaluation that limits the parameters between bounds.
See (Harms 2017) for more information on this topic.
You can define the transformation function used by setting the ``parameter_transform`` attribute.
For an overview of the available parameter transformations, see :mod:`~mot.model_building.parameter_functions.transformations` in MOT.

Sampling
$$$$$$$$
For sampling one needs to define per parameter a prior and a proposal function.
These can easily be added in MDT using the attributes ``sampling_proposal`` and ``sampling_prior``.
Additionally, one can define a sampling statistic function ``sampling_statistics``.
This function is run after sampling and should return basic statistics on the observed samples.
For most parameters this uses the Gaussian distribution to return a mean and standard deviation.
For angular parameters it uses the circular Gaussian distribution.


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


.. _dynamic_modules_compartments:


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


.. _dynamic_modules_compartments_extra_result_maps:

Extra result maps
^^^^^^^^^^^^^^^^^
It is possible to add additional parameter maps to the fitting and sampling results.
These maps are meant to be forthcoming to the end user by providing additional maps of interest to the output.
By adding additional maps to a compartment one ensures that all composite models that use that compartment profit from the additionally calculated maps.
One can also add additional output maps to the composite models, but they do not have this advantage.
Preferably one adds the additional maps to the compartment model.
If that does not work because you need information from more than one compartment you can place the additional map computations in the composite model.

In compartments, one can add extra/additional result maps by adding the bound function ``get_extra_result_maps`` to your compartment. As an example:

.. code-block:: python

    ...
    from mdt.components_loader import bind_function

    class Stick(CompartmentConfig):
        ...
        @bind_function
        def get_extra_results_maps(self, results_dict):
            return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                                results_dict[self.name + '.phi'])


In this example we added the (x, y, z) component vector to the results for the Stick compartment.


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
MDT parses this string, loads the compartments from the compartment models and uses the CL code of these compartments to create the CL objective function for your complete composite model.

The example above combines the compartments (``Ball`` and ``Stick``) as a weighted summation using the special compartment ``Weight`` for the compartment weighting
(these weights are sometimes called volume fractions).
The example also shows compartment renaming.
Since it is possible to use a compartment multiple times, it is necessary to rename the double compartments to ensure that all the compartments have a unique name.
This renaming can be done by specifying the renamed model name in parenthesis after the compartment model name.
For example ``Stick(Stick0)`` refers to a ``Stick`` compartment that has been renamed to ``Stick0``. This new name is then used to refer to that specific compartment in the
rest of the composite model attributes.

The composite models have more functionality than what is shown here. For example, they support parameter dependencies, initialization values, parameter fixations and protocol options.
The important functionality is explained here.


Parameter dependencies
^^^^^^^^^^^^^^^^^^^^^^
Parameter dependencies make explicit the dependency of one parameter on another.
For example, some models have both an intra- and an extra-axonal compartment that both feature the ``theta`` and ``phi`` fibre orientation parameters.
It could be desired that these angles are exactly the same for both compartments and that they both reflect the exact same fibre orientation.
One possibility to solve this would be to create a new compartment having the features of both the intra- and the extra-axonal compartment.
This however lowers the reusability of the compartments.
Instead, one could define parameter dependencies in the composite model, for example:

.. code-block:: python

    ...
    from mot.model_building.parameter_functions.dependencies import SimpleAssignment

    class NODDI(DMRICompositeModelConfig):
        ...
        dependencies = (
            ('NODDI_EC.theta', SimpleAssignment('NODDI_IC.theta')),
            ('NODDI_EC.phi', SimpleAssignment('NODDI_IC.phi'))
        )


In this example, we added the attribute ``dependencies`` to our composite model (in this example, NODDI).
This attribute accepts a list of tuples, with as first elements the name of the parameter that is being locked to a dependency and second an dependency object.
In this case we added two simple assignment dependencies in which the theta and phi of the NODDI_EC compartment are locked to that of the NODDI_IC compartment.
Hence, this also removes the NODDI_EC theta and phi from the list of parameters to optimize, reducing degrees of freedom in the model.


Default Weights dependency
^^^^^^^^^^^^^^^^^^^^^^^^^^
Most composite models consist of a weighted sum of compartments models.
An implicit dependency in this set-up is that those weights must exactly sum to one.
To ensure this, MDT adds by default a dependency to the last Weight compartment in the composite model definition
(see the section above on parameter dependencies in general).
This dependency first normalizes the n-1 Weight compartments by their sum :math:`s = \sum_{i}^{n-1}w_{i}` if that sum is larger than one.
The last Weight, not explicitly optimized, is either set to zero, i.e. :math:`w_{n} = 0` or set as :math:`w_{n}=1-s` if s is smaller than zero.

If you wish to disable this feature, for example in a model that does not have a linear sum of Weights, you can use set the attribute ``add_default_weights_dependency`` to false, e.g.:

.. code-block:: python


    class MyModel(DMRICompositeModelConfig):
        ...
        add_default_weights_dependency = False



.. _dynamic_modules_composite_models_protocol_options:


Protocol options
^^^^^^^^^^^^^^^^
It is possible to specify protocol options in a composite model.
These protocol options are meant to allow the composite model to select, using the protocol, only those volumes that it can use for optimization.
For example, the Tensor model is defined to work with b-values up to 1500 s/mm^2, yet the user might be using a dataset that has more shells with some shells above that b-value threshold.
To prevent the user from having to load a separate protocol and dMRI dataset for the Tensor model and another for the other models, we implemented in MDT the model protocol options.
This way, the end user can provide the whole protocol file and the models will pick from that what they need.

To include this mechanism to your composite model, you have to add the bound function ``_get_suitable_volume_indices`` to your model definition. For example:

.. code-block:: python

    ...
    from mdt.components_loader import bind_function

    class Tensor(DMRICompositeModelConfig):

        ...

        @bind_function
        def _get_suitable_volume_indices(self, problem_data):
            return protocol.get_indices_bval_in_range(start=0, end=1.5e9 + 0.1e9)


The function decorator ``bind_function`` ensures that the function is added to the model constructed from this definition.
To select the volumes you wish to use the function is given the current :ref:`concepts_problem_data_models`.
This function should then return a list of integers specifying the volumes (and therefore protocol rows) you wish to use in the analysis of this model.
To use all volumes you can use something like this:

.. code-block:: python

    @bind_function
    def _get_suitable_volume_indices(self, problem_data):
        return list(range(problem_data.protocol.length))


Post optimization modifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Post optimization modifiers allow you to change the values of parameter maps after optimization, and allow you to add new maps to the final results.
These modifiers complement the :ref:`dynamic_modules_compartments_extra_result_maps` from the compartment models.
An example can be found in the CHARMED model, where one by default expects the ``FR`` map to be returned from model fitting.
Since FR is not a parameter of any of the compartments, it would normally not be returned.
To prevent the end users from having to do additional post-processing to add this map themselves,
we added in MDT a post optimization modifier that adds the FR map automatically after optimization:

.. code-block:: python

    class CHARMED_r3(DMRICompositeModelConfig):
        ...
        post_optimization_modifiers = [
            ('FR', lambda results: 1 - results['w_hin0.w'])
        ]

Here FR is defined as :math:`1 - w_{hin_{0}}`, which is the same as :math:`\sum_{i}^{n} w_{res_{i}}`.

More in general, for every additional map you wish to add in a model, add a tuple with the name of the desired map
and as value a function callback that accepts the current dictionary with result maps and returns a new map to add to this dictionary.


Evaluation function and likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Models are optimized by finding the set of free parameter values :math:`x \in R^{n}` that minimize the evaluation function or objective function of the
modeling errors :math:`(O − S(x))` with :math:`O` the observed data and :math:`S(x)` the model signal estimate.
In diffusion MRI the common likelihood models are the *Gaussian*, *Rician* and *Offset-Gaussian*.
Each has different characteristics and implements the modeling :math:`(O − S(x))` in a slightly different way.
Following (Harms 2017) we use, by default, the Offset Gaussian likelihood model for all models.
To change this to another likelihood model for one of your models you can override the ``evaluation_model`` attribute, for example:

.. code-block:: python

    ...
    from mot.model_building.evaluation_models import RicianEvaluationModel

    class MyModel(DMRICompositeModelConfig)
        ...
        evaluation_model = RicianEvaluationModel()


Please note though that the Rician evaluation model is not very stable numerically.


Cascade models
--------------
Cascade models are meant to make chained optimization procedures explicit.
For example, complex models like CHARMED and NODDI are optimized better if the optimization routine is initialized at a better starting point.
This could be as simple as initializing the model with the height of the unweighted signal, or be as complex as initializing the fibre directions and volume fractions.
To create a new cascade model, you will need to specify, at a minimum, the ``name`` and ``models`` attribute of the cascade definition, for example:

.. code-block:: python

    class CHARMED_r3(CascadeConfig):

        name = 'CHARMED_r3 (Cascade)'
        models = ('BallStick_r3 (Cascade)',
                  'CHARMED_r3')


In this example we create a cascade going from a (cascaded) BallStick_r3 model to a CHARMED_r3 model.
See the next section for more on the initializations.


Parameter initializations
^^^^^^^^^^^^^^^^^^^^^^^^^
Identical parameters in between cascade steps are initialized automatically.
That is, in the previous example the ``S0.s0`` parameter is initialized automatically from the BallStick_in3 results the to CHARMED_r3 model since
both the CHARMED_r3 and the BallStick_r3 model have a S0 compartment with a s0 parameter.

Using the attribute ``inits`` you can provide an additional set of parameter initializations to add to or overwrite the default implicit initializations.
Extending the previous CHARMED_r3 example, we have:

.. code-block:: python

    class CHARMED_r3(CascadeConfig):
        ...
        inits = {'CHARMED_r3': [('Tensor.theta', 'Stick0.theta'),
                                ('Tensor.phi', 'Stick0.phi'),
                                ('w_res0.w', 'w_stick0.w'),
                                ('w_res1.w', 'w_stick1.w'),
                                ('w_res2.w', 'w_stick2.w'),
                                ('CHARMEDRestricted0.theta', 'Stick0.theta'),
                                ('CHARMEDRestricted0.phi', 'Stick0.phi'),
                                ...
                                ]}

In this extended example we still automatically initialize the S0 compartment and additionally initialize a lot more parameters.
These ``inits`` should be read as: "When optimizing CHARMED_r3, take from the previous model fit the 'Stick0.theta' results and use that to initialize the 'Tensor.theta' parameter.
Then, take the 'Stick0.phi results and use that to ... and so forth ...".


Parameter fixations
^^^^^^^^^^^^^^^^^^^
It is also possible to specify parameter fixations in between cascade steps.
These fixations fix the appointed parameter to a specific value, removing that parameter from the list of optimized functions.
This reduces the degrees of freedom of the optimized model which normally leads to faster optimization times and possibly better results.
For example:

.. code-block:: python

    class CHARMED_r3_Fixed(CascadeConfig):
        ...
        fixes = {'CHARMED_r3': [('CHARMEDRestricted0.theta', 'Stick0.theta'),
                                ('CHARMEDRestricted0.phi', 'Stick0.phi'),
                                ...
                                ]}


Using the attribute ``fixes`` we here specified that some of the parameters are fixed to a previous value instead of initializing them.
In this example we fixed the ``theta`` and ``phi`` parameter of the intra-axonal compartments to that of a previous BallStick fit, which means we are no longer optimizing
those directions but take them literally from the previous model.


Value specification syntax
^^^^^^^^^^^^^^^^^^^^^^^^^^
There are various ways in which it is possible to specify the value to use for the initialization or fixation of the next model in the cascade.
The basic syntax of the ``inits`` and ``fixes`` attribute is:

.. code-block:: python

    {<model_name>: [(<param_name>, <value_specification>), ... ],
     ...
    }

Where model name is one of the models in the cascade followed by a list of parameters value specifications that specify what to do with the parameters of that model.
There are three different parameter specifications possible:

* *Single value* or *ndarray*: specify a value to use, for example using a value of 1e5 for the S0.s0 parameter of a model
* *String*: the name of a parameter from the previous model, this is the most common approach
* *Function*: specify a function that accepts two dictionaries, ``output_previous`` and ``output_all_previous``.
  The first contains the results of the previous model fit indexed by parameter names.
  The second contains the results of all prior model estimates, indexed first by model name and second by parameter name.

An example highlighting these syntactic options would be:

.. code-block:: python

    class Example(CascadeConfig):
        ...
        models = ('S0',
                  'BallStick_r1',
                  'NODDI')

        inits = {'BallStick_r1': [('S0.s0', 1e5)],
                 'NODDI': [('NODDI_IC.theta', 'Stick.theta'),
                           ('NODDI_IC.phi', lambda output_previous, output_all_previous: output_previous['Stick.phi']]),
                           ('S0.s0', lambda output_previous, output_all_previous: output_all_previous['S0']['S0.s0'])]
                }


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
