.. _dynamic_modules_library_functions:

*****************
Library functions
*****************
Library functions are Python wrappers around reusable CL functions.
Having specified them in the compartment model they are included in the CL kernel and hence their CL functions are usable from within the compartment models.

To create one, please look at one of the existing library functions.
To use one, add the attribute ``dependency_list`` to your compartment model's definition and add there one or more of the library functions to include in the CL code.

Next to the library functions available in MDT, there are also a few loadable functions in MOT, see :mod:`~mot.library_functions` for a list.
