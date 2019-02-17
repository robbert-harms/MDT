.. _dynamic_modules_composite_models:

****************
Composite models
****************
The composite models, or, multi-compartment models, are the models that MDT actually optimizes.
Composite models are formed by a combination / composition, of compartment models.

When asked to optimize (or sample) a composite model, MDT combines the CL code of the compartments into one objective function and
combines it with a likelihood function (Rician, OffsetGaussian, Gaussian).
Since the compartments already contain the CL code, no further CL modeling code is necessary in the multi-compartment models.

Composite models are defined by inheriting from :class:`~mdt.component_templates.composite_models.CompositeModelTemplate`.
The following is an minimal example of a composite (multi-compartment) model in MDT::

    class BallStick_r2(CompositeModelTemplate):

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
This renaming is done by specifying the nickname in parenthesis after the compartment.
For example ``Stick(Stick0)`` refers to a ``Stick`` compartment that has been renamed to ``Stick0``.
This new name is then used to refer to that specific compartment in the rest of the composite model attributes.

The composite models have more functionality than what is shown here.
For example, they support parameter dependencies, initialization values, parameter fixations and protocol options.


Parameter dependencies
======================
Parameter dependencies make explicit the dependency of one parameter on another.
For example, some models have both an intra- and an extra-axonal compartment that both feature the ``theta`` and ``phi`` fibre orientation parameters.
It could be desired that these angles are exactly the same for both compartments, that is, that they both reflect the exact same fibre orientation.
One possibility to solve this would be to create a new compartment having the features of both the intra- and the extra-axonal compartment.
This however lowers the reusability of the compartments.
Instead, one could define parameter dependencies in the composite model.
For example:

.. code-block:: python

    class NODDI(CompositeModelTemplate):
        ...
        fixes = {
            ...
            'Ball.d': 3.0e-9,
            'NODDI_EC.dperp0': 'NODDI_EC.d * (w_ec.w / (w_ec.w + w_ic.w))',
            'NODDI_EC.kappa': 'NODDI_IC.kappa',
            'NODDI_EC.theta': 'NODDI_IC.theta',
            'NODDI_EC.phi': 'NODDI_IC.phi'
        }


In this example, we used the attribute ``fixes`` to specify dependencies and parameter fixations.
The attribute ``fixes`` accepts a dictionary with as key the name of the parameter and as value a scalar, a map or a dependency.
The dependency can either be given as a string or as a dependency object.

In the example above we added two simple assignment dependencies in which the theta and phi of the NODDI_EC compartment are locked to that of the NODDI_IC compartment.
This dependency locks the NODDI_EC theta and phi to that of NODDI_IC assuring that both the intra cellular and extra cellular models reflect the same orientation.


Weights sum to one
==================
Most composite models consist of a weighted sum of compartments models.
An implicit dependency in this set-up is that those weights must exactly sum to one.
To ensure this, MDT adds, by default, a dependency to the last Weight compartment in the composite model definition.
This dependency first normalizes (if needed) the n-1 Weight compartments by their sum :math:`s = \sum_{i}^{n-1}w_{i}`.
Then, the last Weight, which is not optimized explicitly, is then either set to zero, i.e. :math:`w_{n} = 0` or set as :math:`w_{n}=1-s` if s is smaller than zero.

If you wish to disable this feature, for example in a model that does not have a linear sum of weighted compartments, you can use set the attribute ``enforce_weights_sum_to_one`` to false, e.g.:

.. code-block:: python


    class MyModel(CompositeModelTemplate):
        ...
        enforce_weights_sum_to_one = False


.. _dynamic_modules_composite_models_protocol_options:


Protocol options
================
It is possible to add dMRI volume selection to a composite model using the "protocol options".
These protocol options allow the composite model to select, using the protocol, only those volumes that it can use for optimization.
For example, the Tensor model is defined to work with b-values up to 1500 s/mm^2, yet the user might be using a dataset that has more shells, with some shells above the b-value threshold.
To prevent the user from having to load a separate dataset for the Tensor model and another dataset for the other models, we implemented in MDT model protocol options.
This way, the end user can provide the whole protocol file and the models will pick from it what they need.

Please note that these volume selections only work with columns in the protocol, not with the ``extra_protocol`` maps.

There are two ways to enable this mechanism in your composite model.
The first is to add the ``volume_selection`` directive to your model:

.. code-block:: python

    class Tensor(CompositeModelTemplate):
        ...
        volume_selection = {'b': [(0, 1.5e9 + 0.1e9)]}


This directive specifies that we wish to use a subset of the weighted volumes, that is, a single b-value range with b-values between b=0 and b=1.5e9 s/m^2.
Each key in ``volume_selection`` should refer to a column in the protocol file and each value should be a list of ranges.

The second method is to add the bound function ``_get_suitable_volume_indices`` to your model definition. For example:

.. code-block:: python

    ...
    from mdt.component_templates.base import bind_function

    class Tensor(CompositeModelTemplate):
        ...

        @bind_function
        def _get_suitable_volume_indices(self, input_data):
            return protocol.get_indices_bval_in_range(start=0, end=1.5e9 + 0.1e9)


This function should then return a list of integers specifying the volumes (and therefore protocol rows) you wish to use in the analysis of this model.
To use all volumes you can use something like this:

.. code-block:: python

    @bind_function
    def _get_suitable_volume_indices(self, input_data):
        return list(range(input_data.protocol.length))


.. _dynamic_modules_composite_models_extra_result_maps:

Extra result maps
=================
It is also possible to add additional parameter maps to the fitting and sampling results.
These maps are meant to be forthcoming to the end-user by providing additional maps to the output.
Extra results maps can be added by both the composite model as well as by the compartment models.

Just as with compartment models, one can add extra output maps to the optimization results and to the sampling results as:

.. code-block:: python

    class MyModel(CompositeModelTemplate):
        ...
        extra_optimization_maps = [
            lambda results: ...
        ]

        extra_sampling_maps = [
            lambda samples: ...
        ]

where each callback function should return a dictionary with extra maps to add to the output.


.. _dynamic_modules_composite_model_likelihood_function:

Likelihood functions
====================
Models are optimized by finding the set of free parameter values :math:`x \in R^{n}` that minimize the likelihood function of the
modeling errors :math:`(O - S(x))` with :math:`O` the observed data and :math:`S(x)` the model signal estimate.
In diffusion MRI the common likelihood models are the *Gaussian*, *Rician* and *OffsetGaussian* models.
Each has different characteristics and implements the modeling :math:`(O - S(x))` in a slightly different way.
Following (Harms 2017) we use, by default, the Offset Gaussian likelihood model for all models.
To change this to another likelihood model for one of your models you can override the ``likelihood_function`` attribute, for example:

.. code-block:: python

    class MyModel(CompositeModelTemplate)
        ...
        likelihood_function = 'Rician'


By default the ``likelihood_function`` attribute is set to ``OffsetGaussian``.
The likelihood function can either be defined as a string or as an object.
Using a string, the possible options are ``Gaussian``, ``OffsetGaussian`` and ``Rician``.
Using an object, you must provide an instance of :class:`mdt.model_building.likelihood_functions.LikelihoodFunction`.
For example:

.. code-block:: python

    ...
    from mdt.model_building.likelihood_functions import RicianLikelihoodFunction

    class MyModel(CompositeModelTemplate)
        ...
        likelihood_function = RicianLikelihoodFunction()


All listed likelihood functions require a standard deviation :math:`\sigma` representing the noise in the input data.
This value is typically taken from the noise of the images in the complex domain and is provided in the input data (see :ref:`concepts_input_data_models`).


Constraints
===========
It is possible to add additional inequality constraints to a composite model, using the ``constraints`` attribute.
These constraints need to be added as the result of the function :math:`g(x)` where we assume :math:`g(x) \leq 0`.

For example, in the NODDIDA model we implemented the constraint that the intra-cellular diffusivity must be larger than the extra-cellular diffusivity, following Kunz et al., NeuroImage 2018.
Mathematically, this constraint can be stated as :math:`d_{ic} \geq d_{ec}`. For implementation in MDT, we will state it as :math:`d_{ec} - d_{ic} \leq 0` and implement it as::

    class NODDIDA(CompositeModelTemplate)
        ...
        constraints = '''
            constraints[0] = NODDI_EC.d - NODDI_IC.d;
        '''

This ``constraints`` attribute can hold arbitrary OpenCL C code, as long as it contains the literal ``constraints[i]`` for each additional constraint ``i``.

From this constraints string, MDT creates a function with the same dependencies and parameters as the composite model.
This function is then provided to the optimization routines, which enforce it using the *penalty* method (https://en.wikipedia.org/wiki/Penalty_method).
