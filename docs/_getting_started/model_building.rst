**************
Model building
**************
This section explains how new models can be implemented in MDT.
In MDT, models are constructed in an object oriented fashion with more complex objects being constructed out of simpler parts.
The following figure shows the order of model construction in MDT:

.. image:: _static/figures/mdt_model_building.png

As shown in the figure, compartments models are constructed using one or more parameters.
Composite models in turn are built out of one or more compartment models.
Cascade models in turn consist out of one or more composite models.

Adding new components to MDT is made easy using the :ref:`dynamic_modules` in which new components can be defined just by adding Python script files to the MDT components folder in your home directory,
found at ``~/.mdt/<version>/components/`` where ``~`` stands for your home folder.
An general advice would be to, where possible, adapt existing components by copying them from the ``standard`` folder to the ``user`` folder and adapt them to the specification or your new model.

While the sections in this chapter follow the order of the model dependencies as shown in the figure above, defining a new composite model is conceptually usually done in the reversed order.
For example, one usually defines a new model mathematically to later split it up in separate compartments and distinct parameters.
This chapter is meant as both a general reading as well as a reference manual for constructing new models.


Defining new Parameters
=======================
Before defining a new parameter please have a look at the available parameters in your home folder at ``~/.mdt/<version>/components/standard/parameters/`` (where ``~`` stands for your home folder).
In this folder you will find multiple ``.py`` files each containing multiple parameter definitions.
While MDT is indifferent to the names of these files, the names give an indication of the type of parameters that can be found in any of the given files.
For example, the file ``free.py`` contains all free parameters used in the standard compartment models in MDT.

To add a new parameter model, create a blank ``.py`` folder in your users parameter folder and add to that your new parameter(s).
MDT requires that every parameter has its own unique name to prevent confusion, so please make sure you are using an unique name for your parameter (else an error will be raised by MDT).

For an overview of the available options for configuring your parameter please see :ref:`dynamic_modules_parameters`.

To check if a parameter can be found by MDT, you can use the following code in a Python shell:

.. code-block:: python

    >>> import mdt
    >>> mdt.load_component('parameters', '<param_name>')

Where ``<param_name>`` should be substituted by the name of your parameter.
If that works without errors your parameter can be found and can be used inside compartment models.


.. _model_building_defining_compartments:

Defining new Compartments
=========================
As with the parameters, new compartments can be added by adding small Python script files to the ``user`` folder in your MDT components folder.

In contrast to the parameters, the filenames of the compartment models is important and should match exactly the class name of your implemented model.
For example, the ``Stick`` compartment model can be found in a file named ``Stick.py`` with a matching class name:

``Stick.py``:

.. code-block:: python

    class Stick(CompartmentConfig):
        ...


Compartment names in MDT have to be unique, so when adapting an old compartment please rename it to a new unique name (don't forget to rename the filename as well).

For an overview of the available options for configuring your compartment model please see :ref:`dynamic_modules_compartments`.

To check if a compartment can be found by MDT, you can use the following code in a Python shell:

.. code-block:: python

    >>> import mdt
    >>> mdt.load_component('compartment_models', '<compartment_name>')

Where ``<compartment_name>`` should be substituted by the name of your compartment.
If that works without errors your compartment can be found and can be used inside composite models.


Defining new Composite models
=============================
New composite models can be defined in any ``.py`` file in the ``user/composite_models`` folder in the MDT modules folder on your home drive.
In contrast to the compartment models, the composite models do not need to have their own file and multiple composite models can be defined in the same Python file.
The same as with the other modules, the composite models need to have unique names else an error will be raised by MDT.

For an overview of the available options for configuring your composite model please see :ref:`dynamic_modules_composite_models`.

To check if a composite model can be found by MDT, you can use the following code in a Python shell:

.. code-block:: python

    >>> import mdt
    >>> mdt.get_model('<model_name>')

Where ``<model_name>`` should be substituted by the name of your composite model, e.g. 'NODDI' or 'BallStick'.
If that works without errors your composite model can be found and can be used for model fitting.


Defining new Cascade models
===========================
The same as with the composite models, cascade models can be defined simply by adding a Python text file to your ``user/cascade_models`` folder.
As with the composite models, the name of the containing Python file does not matter but the cascade models need to be unique.

The general naming guideline is that the cascade model is named after the last model in the cascade, with the addition of the suffix ``(Cascade)`` to the model.
So, for example, the cascade for the ``NODDI`` composite model would be named ``NODDI (Cascade)``.
Alterations on the general cascade can be named by adding keywords after to the Cascade suffix.
For example, in MDT, cascades with parameter fixations are often indicated by ``... (Cascade|fixed)``.

For an overview of the available options for configuring your composite model please see :ref:`dynamic_modules_cascades`.

To check if a cascade model can be found by MDT, you can use the following code in a Python shell:

.. code-block:: python

    >>> import mdt
    >>> mdt.get_model('<model_name>')

Where ``<model_name>`` should be substituted by the name of your composite model, e.g. 'NODDI (Cascade)' or 'CHARMED (Cascade|fixed)'.
If that works without errors your composite model can be found and can be used for model fitting.
