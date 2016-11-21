.. _dynamic_modules_composite_models:

****************
Composite models
****************
The composite models, or, multi-compartment models, are the models that MDT actually optimizes.
Just as the compartments are built using parameters as a building block, the composite models are built using compartments as a building block.
Since the compartments already contain the CL code, no further model coding is necessary in the multi-compartment models.
When asked to optimize (or sample) a model, MDT combines the CL code of the compartments into one objective function and uses the parameters of the compartments to load the correct data.

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

The example above also shows compartment renaming.
Since it is possible to use a compartment multiple times, it is necessary to rename the double compartments to ensure that all the compartments have a unique name.
This renaming can be done by specifying the renamed model name in parenthesis after the compartment model name.
For example ``Stick(Stick0)`` refers to a ``Stick`` compartment that has been renamed to ``Stick0``.
This new name is then used to refer to that specific compartment in the rest of the composite model attributes.

The composite models have more functionality than what is shown here.
For example, they support parameter dependencies, initialization values, parameter fixations and protocol options.
The important functionality is explained here.


Parameter dependencies
======================
Parameter dependencies make explicit the dependency of one parameter on another.
For example, some models have both an intra- and an extra-axonal compartment that both feature the ``theta`` and ``phi`` fibre orientation parameters.
It could be desired that these angles are exactly the same for both compartments, that is, that they both reflect the exact same fibre orientation.
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
            ('NODDI_EC.phi', 'NODDI_IC.phi')
        )


In this example, we added the attribute ``dependencies`` to our composite model (in this example, NODDI).
This attribute accepts a list of tuples, with as first elements the name of the parameter that is being locked to a dependency and second an dependency object or a string.
If a string is given as an dependency we will use that as a SimpleAssignment object.
In the example shown above we added two simple assignment dependencies in which the theta and phi of the NODDI_EC compartment are locked to that of the NODDI_IC compartment.
This automatically removes the NODDI_EC theta and phi from the list of parameters to optimize, reducing the degrees of freedom of the model.


Default Weights dependency
==========================
Most composite models consist of a weighted sum of compartments models.
An implicit dependency in this set-up is that those weights must exactly sum to one.
To ensure this, MDT adds by default a dependency to the last Weight compartment in the composite model definition.
This dependency first normalizes the n-1 Weight compartments by their sum :math:`s = \sum_{i}^{n-1}w_{i}` if that sum is larger than one.
The last Weight, not explicitly optimized, is then either set to zero, i.e. :math:`w_{n} = 0` or set as :math:`w_{n}=1-s` if s is smaller than zero.

If you wish to disable this feature, for example in a model that does not have a linear sum of Weights, you can use set the attribute ``add_default_weights_dependency`` to false, e.g.:

.. code-block:: python


    class MyModel(DMRICompositeModelConfig):
        ...
        add_default_weights_dependency = False



.. _dynamic_modules_composite_models_protocol_options:


Protocol options
================
It is possible to add a sort of dMRI volume selection to a composite model using the "protocol options".
These protocol options allow the composite model to select, using the protocol, only those volumes that it can use for optimization.
For example, the Tensor model is defined to work with b-values up to 1500 s/mm^2, yet the user might be using a dataset that has more shells with some shells above that b-value threshold.
To prevent the user from having to load a separate dataset for the Tensor model and another dataset for the other models, we implemented in MDT model protocol options.
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
===========================
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

More in general, for every additional map you wish to add, add a tuple with the name of the desired map
and as value a function callback that accepts the current dictionary with result maps and returns a new map to add to this dictionary.


.. _dynamic_modules_composite_model_evaluation_function:

Evaluation function and likelihood
==================================
Models are optimized by finding the set of free parameter values :math:`x \in R^{n}` that minimize the evaluation function or objective function of the
modeling errors :math:`(O - S(x))` with :math:`O` the observed data and :math:`S(x)` the model signal estimate.
In diffusion MRI the common likelihood models are the *Gaussian*, *Rician* and *Offset-Gaussian*.
Each has different characteristics and implements the modeling :math:`(O - S(x))` in a slightly different way.
Following (Harms 2017) we use, by default, the Offset Gaussian likelihood model for all models.
To change this to another likelihood model for one of your models you can override the ``evaluation_model`` attribute, for example:

.. code-block:: python

    ...
    from mot.model_building.evaluation_models import RicianEvaluationModel

    class MyModel(DMRICompositeModelConfig)
        ...
        evaluation_model = RicianEvaluationModel()


Please note though that the Rician evaluation model is not very stable numerically.

Most evaluation functions require a standard deviation :math:`\sigma` of the noise of the images in the complex domain.
This standard deviation is, during analysis, taken from the :ref:`concepts_problem_data_models`.
