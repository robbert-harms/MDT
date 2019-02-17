.. _dynamic_modules_compartments:

******************
Compartment models
******************
The compartment models are the building blocks of the composite models.
They consists in basis of two parts, a list of parameters (see :ref:`dynamic_modules_parameters`) and the model code in OpenCL C (see :ref:`concepts_cl_code`).
At runtime, MDT loads the C/CL code of the compartment model and combines it with the other compartments to form the composite model.

Compartment models can be defined using the templating mechanism by inheriting from :class:`~mdt.component_templates.compartment_models.CompartmentTemplate`.
For example, the Stick model can be defined as::

    from mdt.component_templates.compartment_models import CompartmentTemplate

    class Stick(CompartmentTemplate):

        parameters = ('g', 'b', 'd', 'theta', 'phi')
        cl_code = '''
            float4 n = (float4)(cos(phi) * sin(theta),
                                                  sin(phi) * sin(theta),
                                                  cos(theta),
                                                  0);

            return exp(-b * d * pown(dot(g, n), 2));
        '''


This ``Stick`` example contains all the basic definitions required for a compartment model: a parameter list and CL code.


Defining parameters
===================
The elements of the parameter list can either be string referencing one of the parameters in the library (like shown in the example above),
or it can be a direct instance of a parameter. For example, this is also a valid parameter list::

    class special_param(FreeParameterTemplate):
        ...

    class MyModel(CompartmentTemplate):

        parameters = ('g', 'b', special_param()())
        ...


where the parameters ``g`` and ``b`` are loaded from the dynamically loadable parameters while the ``special_param`` is given as a parameter instance.
It is also possible to provide a nickname for a parameter by stating something like::

    parameters = ('my_theta(theta)', ...)

Here, the parameter ``my_theta`` is loaded with the nickname ``theta``.
This allows you to use simpler names for the parameters of a compartment and allows you to swap a parameter for a different type while still using the same (external) name.


Dependency list
===============
Some models may depend on other compartment models or on library functions.
These dependencies can be specified using the ``dependencies`` attribute of the compartment model definition.
As an example::

    dependencies = ('erfi', 'MRIConstants', 'CylinderGPD')

This list should contain strings with references to either library functions or other compartment models.
In this example the ``erfi`` library function is loaded from MOT, ``MRIConstants`` from MDT and ``CylinderGPD`` is another compartment model which our example depends on.

Adding items to this list means that the corresponding CL functions of these components are included into the optimized OpenCL kernel and allows you to use the corresponding CL code in your compartment model.

For example, in the dependency list above, the ``MRIConstants`` dependency adds multiple constants to the kernel,
like ``GAMMA_H``, the gyromagnetic ratio of in the nucleus of H in units of (rad s^-1 T^-1).
By adding ``MRIConstants`` as a compartment dependency, this constant can now be used in your compartment model function.


Defining extra functions for your code
======================================
It is possible that a compartment model needs some auxiliary functions that are too small for an own library function.
These can be added to the compartment model using the ``cl_extra`` attribute. For example::

    class MyCompartment(CompartmentTemplate):

        parameters = ('g', 'b', 'd')
        cl_code = 'return other_function(g, b, d);'
        cl_extra = '''
            double other_function(
                    float4 g,
                    mot_float_type b,
                    mot_float_type d){

                ...
            }
        '''


.. _dynamic_modules_compartments_extra_result_maps:


Extra result maps
=================
It is possible to add additional parameter maps to the fitting and sampling results.
These maps are meant to be forthcoming to the end-user by providing additional maps to the output.
Extra results maps can be added by both the composite model as well as by the compartment models.
By defining them in a compartment model one ensures that all composite models that use that compartment profit from the additional output maps.

Just as with composite models, one can add extra output maps to the optimization results and to the sampling results as:

.. code-block:: python

    class MyCompartment(CompartmentTemplate):
        ...
        extra_optimization_maps = [
            lambda results: ...
        ]

        extra_sampling_maps = [
            lambda samples: ...
        ]

where each callback function should return a dictionary with extra maps to add to the output.


Constraints
===========
It is possible to add additional inequality constraints to a compartment model, using the ``constraints`` attribute.
These constraints need to be added as the result of the function :math:`g(x)` where we assume :math:`g(x) \leq 0`.

For example, in the Tensor model we implemented the constraint that the diffusivities must be in a strict order, such that
:math:`d_{\parallel} \geq d_{\perp_{0}} \geq d_{\perp_{1}}`.

For implementation in MDT, we will state this as the two constraints :math:`d_{\perp_{0}} \leq d_{\parallel}` and :math:`d_{\perp_{1}} \leq d_{\perp_{0}}`, and implement it as::

    class Tensor(CompartmentTemplate)
        ...
        constraints = '''
            constraints[0] = dperp0 - d;
            constraints[1] = dperp1 - dperp0;
        '''

This ``constraints`` attribute can hold arbitrary OpenCL C code, as long as it contains the literal ``constraints[i]`` for each additional constraint ``i``.

From this constraints string, MDT creates a function with the same dependencies and parameters as the compartment model.
This function is then provided to the optimization routines, which enforce it using the *penalty* method (https://en.wikipedia.org/wiki/Penalty_method).
