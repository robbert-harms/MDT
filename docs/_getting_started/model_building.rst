**************
Model building
**************
This section explains how new models can be added to MDT.
In MDT, models are constructed in an object oriented fashion with more complex objects being constructed out of simpeler parts.
The following figure shows the order of model construction in MDT.

.. image:: _static/figures/mdt_model_building.png

As shown in the figure, compartments models are constructed using one or more parameters.
The compartment models in turn form the basis of the composite models which in turn form the cascade models.

Adding new features such as models to MDT is made easy using the :ref:`dynamic_modules` in which new components can be defined just by adding new text files to a folder in your home directory.
The general guideline in this chapter is to copy interesting components from the ``standard`` MDT folder to your own ``user`` folder and adapt them to the specification or your new model.
This is of course only necessary for components that you will adapt.

While the sections in this chapter follow the order of the model dependencies as shown in the figure above, defining a new composite model is usually done in the reversed order,
at least conceptually.
For example, one usually defines a new model mathematically and moves from there to separate compartments and parameters.
This chapter is therefore meant as both a general reading as well as a reference manual for constructing new models.


Defining new Parameters
=======================
Before defining a new parameter please have a look at the available parameters in your home folder at ``~/.mdt/<version>/components/standard/parameters/``.
In this folder you will find multiple ``.py`` files each containing multiple parameter definitions.
While MDT is indifferent to the names of these files, the names give an indication of the type of parameters that can be found in any of the given files.
For example, the file ``free.py`` contains all free parameters used in the standard compartment models in MDT.

To add a new compartment model, copy one of the existing ``.py`` files from the ``standard/parameters`` folder to the ``user/parameters`` folder.
Or, alternatively add a blank ``.py`` folder to the users parameters folder.

Having copied one of the existing files make sure you remove all the parameters except those you wish to modify.
Afterwards rename the parameters you want to adapt to a new unique name.
MDT requires that every parameter has its own unique name to prevent confusion.

At this point you can adapt the parameter to your own wishes.
For an overview of the available options please see :ref:`dynamic_modules_parameters`.


.. _model_building_defining_compartments:

Defining new Compartments
=========================



Defining new Composite models
=============================


Defining new Cascade models
===========================
