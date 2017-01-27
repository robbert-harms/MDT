.. _dynamic_modules_cascades:

**************
Cascade models
**************
Cascade models are meant to make chained optimization procedures explicit.
For example, complex models like CHARMED and NODDI are optimized better if the optimization routine is initialized at a better starting point (Harms 2017).
This could be as simple as initializing the model with the height of the unweighted signal, or be as complex as initializing the fibre directions and volume fractions.
To create a new cascade model, you will need to specify, at a minimum, the ``name`` and the ``models`` attributes:

.. code-block:: python

    class CHARMED_r3(CascadeConfig):

        name = 'CHARMED_r3 (Cascade)'
        models = ('BallStick_r3 (Cascade)',
                  'CHARMED_r3')


In this example we create a cascade going from a (cascaded) BallStick_r3 model to a CHARMED_r3 model.


Parameter initializations
=========================
Identical parameters in between cascade steps are initialized automatically.
That is, in the previous example the ``S0.s0`` parameter is initialized automatically from the BallStick_in3 results the to CHARMED_r3 model since
both the CHARMED_r3 and the BallStick_r3 model have a S0 compartment with a s0 parameter.

Using the attribute ``inits`` you can provide an additional set of parameter initializations to add to or overwrite the default implicit initializations.
Extending the previous CHARMED_r3 example, we get:

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
Then, take the 'Stick0.phi results and use that to initialize the 'Tensor.phi' parameter, then ..., and so forth."
For the exact specification syntax, please see below.


Parameter fixations
===================
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
For the exact specification syntax, please see below.


Value specification syntax
==========================
There are various ways in which it is possible to specify the ``inits`` and ``fixes`` in a cascade.
The basic syntax is:

.. code-block:: python

    {<model_name>: [(<param_name>, <value_specification>), ... ],
     ...
    }


This is a dictionary with per model in the cascade you have a list of parameter specifications that specify what to do with the parameters of that model.
There are three different parameter specifications possible:

* *Single value* or *ndarray*: specify a value to use
* *String*: the name of a parameter from the previous model, this is the most common approach
* *Function*: specify a function that accepts two dictionaries, ``output_previous`` and ``output_all_previous``.
  The first contains the results of the previous model fit indexed by parameter names.
  The second contains the results of all prior model estimates, indexed first by model name and second by parameter name.

An example highlighting all these syntactic options would be:

.. code-block:: python

    class Example(CascadeConfig):
        ...
        models = ('S0',
                  'BallStick_r1',
                  'NODDI')

        inits = {'BallStick_r1': [('S0.s0', 1e5)],
                 'NODDI':        [('NODDI_IC.theta', 'Stick.theta'),
                                  ('NODDI_IC.phi', lambda output_previous, output_all_previous:
                                                            output_previous['Stick.phi']),
                                  ('S0.s0', lambda output_previous, output_all_previous:
                                                            output_all_previous['S0']['S0.s0'])]
                }

