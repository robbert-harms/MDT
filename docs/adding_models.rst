.. _adding_models:

#############
Adding models
#############
MDT features a dynamic library that keeps track of all model definitions.
This dynamic library enables defining new models anywhere you like, enabling, for example::

    import mdt
    from mdt.components import get_template

    class Tensor(get_template('composite_models', 'Tensor')):
        likelihood_function = 'Rician'

    mdt.fit_model('Tensor (Cascade)', ...)


Breaking this up, in the first part::

    class Tensor(get_template('composite_models', 'Tensor')):
            likelihood_function = 'Rician'


we load the last available Tensor model template from MDT (using ``get_template('composite_models', 'Tensor')``) and use it as a basis for an updated template.
Then, since this class is also named Tensor (by saying ``class Tensor(...)``) this new template will override the previous Tensor.
The body of this template then updates the previous Tensor, in this case by changing the likelihood function.

In the second part::

    mdt.fit_model('Tensor (Cascade)', ...)

we just call ``mdt.fit_model`` with as model ``Tensor (Cascade)``.
MDT will then load the cascade and its models by taking the last known definitions.
As such, the new ``Tensor`` model with the updated likelihood function will be used in the model fitting.

To remove an entry, you can use, for example:

.. code-block:: python

    mdt.components.remove_last_entry('composite_models', 'Tensor')


************************
Global model definitions
************************
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


The configuration files are discussed in :ref:`configuration`, the components folder are used for housing global model definitions.

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


.. include:: _dynamic_modules/parameters.rst
.. include:: _dynamic_modules/compartment_models.rst
.. include:: _dynamic_modules/composite_models.rst
.. include:: _dynamic_modules/cascade_models.rst
.. include:: _dynamic_modules/library_functions.rst
.. include:: _dynamic_modules/batch_profiles.rst
