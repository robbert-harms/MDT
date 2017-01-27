.. _dynamic_modules_library_functions:

*****************
Library functions
*****************
Library functions are meant for reusable CL functions.
Having specified them in the compartment model they are included in the CL kernel and hence their CL functions are usable from within the compartment models.

They are not used very often and their syntax is less sophisticated as the other components.
To create one, please look at one of the existing library functions.
To use one, add the attribute ``dependency_list`` to your compartment model's definition and add there one or more of the library functions to include in the CL code.

Next to the library functions available in MDT, there are also a few loadable functions in MOT, see :mod:`~mot.model_building.cl_functions.library_functions` for a list.
