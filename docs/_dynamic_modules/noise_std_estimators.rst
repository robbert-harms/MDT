.. _dynamic_modules_noise_std_estimators:

*********************
Noise std. estimators
*********************
There are different strategies possible for calculating the standard deviation needed for any likelihood model in a in a
composite model (see :ref:`dynamic_modules_composite_model_evaluation_function` for more on this).
This standard deviation is normally given by the :ref:`concepts_problem_data_models`, but that value can be set to 'auto' which means one
of these strategies will be used to estimate the noise standard deviation.

The standard deviation estimating modules are simple classes that implement the method ``estimate``.
That method should, given the current problem data, returns a single floating point number for the noise standard deviation.
It is possible to configure which strategies to use and in which order using the MDT configuration option ``noise_std_estimating``,
see the example configuration file for more on this.
