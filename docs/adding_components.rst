.. _components:

#################
Adding components
#################
Components are a modular sub-system of MDT allowing you to add models and other functionality to MDT.

On the moment MDT supports the following components:

* :ref:`dynamic_modules_composite_models`: the models fitted by MDT, these are built out of the compartment models
* :ref:`dynamic_modules_compartments`: reusable models of diffusion and relaxometry models
* :ref:`dynamic_modules_parameters`: definitions of common parameters shared between models
* :ref:`dynamic_modules_cascades`: **deprecated**, initialization definitions for composite models
* :ref:`dynamic_modules_library_functions`: library functions for use in composite models
* :ref:`dynamic_modules_batch_profiles`: for the batch functionality in MDT


.. _components_defining_components:

*******************
Defining components
*******************
There are two ways of adding or updating components in MDT, :ref:`components_global_definitions`, by adding a component to your configuration folder or :ref:`components_dynamic_definitions` by defining it dynamically in your modeling scripts.


.. _components_global_definitions:

Global definitions
==================
For persistent model definitions you can use the ``.mdt`` folder in your home folder.
This folder contains diffusion MRI models and other functionality that you can extend without needing to reinstall or recompile MDT.

The ``.mdt`` folder contains, for every version of MDT that existed on your machine, a directory containing the configuration files and a
folder with the dynamically loadable modules.
A typical layout of the ``.mdt`` directory is:

* .mdt/
    * <version>/
        * mdt.default.conf
        * mdt.conf
        * components/


The configuration files are discussed in :ref:`configuration`, the components folder is used for the global model definitions.

The components folder consists of two sub-folders, *standard* and *user*, with an identical folder structure for the contained modules:

* components/
    * standard
        * compartment_models
        * composite_models
        * ...
    * user
        * compartment_models
        * composite_models
        * ...


By editing the contents of these folders, the user can add, extend and/or remove functionality globally and persistently.
The folder named *standard* contains modules that come pre-supplied with MDT.
These modules can change from version to version and any change you make in in this folder will be lost after an update.
To make persistent changes you can add your modules to the *user* folder.
The content of this folder is automatically copied to a new version.


.. _components_dynamic_definitions:

Dynamic definitions
===================
Alternatively, it is also possible to define components on the fly in your analysis scripts.
This is as simple as defining a template in your script prior to using it in your analysis.
For example, prior to calling the fit model function, you can define a new model as:

.. code-block:: python

    from mdt import CompositeModelTemplate

    class BallZeppelin(CompositeModelTemplate):
        model_expression = '''
            S0 * ( (Weight(w_csf) * Ball) +
                   (Weight(w_res) * Zeppelin) )
        '''

    mdt.fit_model('BallZeppelin', ...)



It is also possible to overwrite existing models on the fly, for example:

.. code-block:: python

    import mdt

    class Tensor(mdt.get_template('composite_models', 'Tensor')):
        likelihood_function = 'Rician'

    mdt.fit_model('Tensor (Cascade)', ...)


Breaking this up, in the first part::

    class Tensor(mdt.get_template('composite_models', 'Tensor')):
            likelihood_function = 'Rician'


we load the last available Tensor model template from MDT (using ``get_template('composite_models', 'Tensor')``) and use it as a basis for an updated template.
Then, since this class is also named Tensor (by saying ``class Tensor(...)``) this new template will override the previous Tensor.
The body of this template then updates the previous Tensor, in this case by changing the likelihood function.

In the second part::

    mdt.fit_model('Tensor (Cascade)', ...)

we just call ``mdt.fit_model`` with as model ``Tensor (Cascade)``.
MDT will then load the cascade and its models by taking the last known definitions.
As such, the new ``Tensor`` model with the updated likelihood function will be used in the model fitting.

To remove an entry, you can use, for example::

    mdt.remove_last_entry('composite_models', 'Tensor')


This functionality allows you to overwrite and add models without adding them to your home folder.


.. include:: _dynamic_modules/composite_models.rst
.. include:: _dynamic_modules/compartment_models.rst
.. include:: _dynamic_modules/parameters.rst
.. include:: _dynamic_modules/cascade_models.rst
.. include:: _dynamic_modules/library_functions.rst
.. include:: _dynamic_modules/batch_profiles.rst
