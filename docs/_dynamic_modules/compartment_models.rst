.. _dynamic_modules_compartments:

******************
Compartment models
******************
The compartment models are the building blocks of the composite models.
They consists in basis of two parts, a list of parameters (see :ref:`dynamic_modules_parameters`) and the model code in OpenCL C (see :ref:`concepts_cl_code`).
At runtime, MDT loads the C code of the compartment model and combines it with the other compartments to form the composite model.

The compartment models must be defined in a ``.py`` file where the **filename matches the class name** and it only allows for **one compartment per file**.
For example, the following example compartment model is named ``Stick`` and must therefore be contained in a file named ``Stick.py``::

    class Stick(CompartmentConfig):

        parameter_list = ('g', 'b', 'd', 'theta', 'phi')
        cl_code = '''
            mot_float_type4 n = (mot_float_type4)(cos(phi) * sin(theta),
                                                  sin(phi) * sin(theta),
                                                  cos(theta),
                                                  0);

            return exp(-b * d * pown(dot(g, n), 2));
        '''


This ``Stick`` example contains all the basic definitions required for a compartment model: a parameter list and CL code.


Defining parameters
===================
The elements of the parameter list can either be string referencing one of the parameters defined in the dynamically loadable parameters (like shown in the example above),
or it can be a direct instance of a parameter. For example, this is also a valid parameter list::

    class special_param(FreeParameterConfig):
        ...

    class MyModel(CompartmentConfig):

        parameter_list = ('g', 'b', special_param())
        ...


where the parameters ``g`` and ``b`` are loaded from the dynamically loadable parameters while the ``special_param`` is given as a parameter instance.


Splitting the CL and Python file
================================
The CL code for a compartment model can either be given in the definition of the compartment, like shown above, or it can be provided in
a separate ``.cl`` file with the same name as the compartment.
An advantage of using an external ``.cl`` file is that you can include additional subroutines in your model definition.
The following is an example of splitting the CL code from the compartment model definition:

``Stick.py``::

    class Stick(CompartmentConfig):

        parameter_list = ('g', 'b', 'd', 'theta', 'phi')

``Stick.cl``:

.. code-block:: c

    double cmStick(
        const mot_float_type4 g,
        const mot_float_type b,
        const mot_float_type d,
        const mot_float_type theta,
        const mot_float_type phi){

        mot_float_type4 n = (mot_float_type4)(cos(phi) * sin(theta),
                                              sin(phi) * sin(theta),
                                              cos(theta),
                                              0);

        return exp(-b * d * pown(dot(g, n), 2));
    }

Note the absence of the attribute ``cl_code`` in the ``Stick.py`` file and note the file naming scheme where the two filenames and the model name are exactly the same.
Also note that with this setup you will need to provide the CL function signature yourself:

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
=================
It is possible to add additional parameter maps to the fitting and sampling results.
These maps are meant to be forthcoming to the end-user by providing additional maps to the output.
Extra results maps can be added by both the composite model as well as by the compartment models.
By defining them in a compartment model one ensures that all composite models that use that compartment profit from the additional output maps.

Just as with composite models, one can add extra output maps by adding a list of post optimization modifiers, like for example:

.. code-block:: python

    from mdt.utils import spherical_to_cartesian

    class Stick(CompartmentConfig):
        ...
        post_optimization_modifiers = [('vec0', lambda results: spherical_to_cartesian(results['theta'], results['phi']))]


In this example we added the (x, y, z) component vector to the results for the Stick compartment.


Dependency list
===============
Some models may depend on other compartment models or on library functions.
These dependencies can be specified using the ``dependency_list`` attribute of the compartment model definition.
As an example:

.. code-block:: python

    from mdt.components_loader import CompartmentModelsLoader

    dependency_list = ('CerfErfi',
                       'MRIConstants',
                       CompartmentModelsLoader().load('CylinderGPD'))

This list should contain :class:`~mot.library_functions.CLLibrary` instances, referencing library functions or other compartment models.
Possible strings in this list are loaded automatically as :ref:`dynamic_modules_library_functions`.
In this example the ``CerfErfi`` library function is loaded from MOT, ``MRIConstants`` from MDT and ``CylinderGPD`` is another compartment model which our example depends on.

Adding items to this list means that the corresponding CL functions of these components are included into the optimized OpenCL kernel
and allows you to use the corresponding CL code in your compartment model.

For example, in the dependency list above, the ``MRIConstants`` dependency adds multiple constants to the kernel,
like for example ``GAMMA_H``, the gyromagnetic ratio of in the nucleus of H in units of (rad s^-1 T^-1).
By adding ``MRIConstants`` as a compartment dependency, this constant can now be used in your compartment model function.
