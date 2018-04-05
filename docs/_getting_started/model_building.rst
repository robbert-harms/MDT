**************
Model building
**************
In MDT, models are constructed in an object oriented fashion with more complex objects being constructed out of simpler parts.
The following figure shows the order of model construction in MDT:

.. image:: _static/figures/mdt_model_building.png
    :align: center

That is, compartments models are constructed using one or more parameters, composite models are built out of one or more compartment models and cascade models consist out of one or more composite models.

In MDT, models are added just by defining them using a templating mechanism.
That is, MDT features a dynamic library system in which components can be overridden by newer versions just by defining the component.
For example, adding a new compartment model or overriding an existing one can be done just by stating:

.. code-block:: python

    class BallStick_r1(CompositeModelTemplate):
        model_expression = '''
            S0 * ( (Weight(w_ball) * Ball) +
                   (Weight(w_stick0) * Stick(Stick0)) )
        '''


In this example we overwrite the existing ``BallStick_r1`` model with a completely new model.
Here, ``CompositeModelTemplate`` tells MDT that this class should be interpreted as a template for a dMRI composite model.
By virtue of meta-classes, this template will automatically be added to the MDT component library for future use.

Using Object Oriented inheritance it is possible to partially rewrite existing models with updated definitions.
For example, instead of defining a completely new ``BallStick_r1`` model, we can also inherit from the existing template::

    from mdt.components import get_template

    class BallStick_r1(get_template('composite_models', 'BallStick_r1')):
        likelihood_function = 'Rician'


Here, we inherit from the existing template and overwrite the likelihood function with Rician.
All other definitions will be taken from the previous template.

See the section :ref:`adding_models` for all details on adding models.

If you have added an interesting model to MDT which you wish to share, please do not hesitate to contact the developers to have it added to MDT.
