########
Concepts
########

This chapter introduces the user to a few concepts that helps to get started with using the Maastricht Diffusion Toolbox.

.. contents:: Table of Contents
   :local:
   :backlinks: none

.. _concepts_protocol:

********
Protocol
********
In MDT, the Protocol contains the settings used for the MRI measurements and is by convention stored in a protocol file with suffix ``.prtcl``.
In such a protocol file, every row represents an MRI volume and every column (tab separated) represents a specific protocol settings.
Since every row (within a column) can have a distinct value, this setup automatically enables *multi-shell protocol* files (just change the b-value per volume/row).

The following is an example of a simple MDT protocol file::

    #gx,gy,gz,b
    0.000e+00    0.000e+00    0.000e+00    0.000e+00
    5.572e-01    6.731e-01    -4.860e-01   1.000e+09
    4.110e-01    -5.254e-01   -7.449e-01   1.000e+09
    ...


And, for a more advanced protocol file::

    #gx,gy,gz,Delta,delta,TE,b
    -0.000e+00  0.000e+00   0.000e+00   2.179e-02   1.290e-02   5.700e-02   0.000e+00
    2.920e-01   1.7100e-01  -9.409e-01  2.179e-02   1.290e-02   5.700e-02   3.000e+09
    -9.871e-01  -8.538e-03  -1.595e-01  2.179e-02   1.290e-02   5.700e-02   5.000e+09
    ...


The header (starting with #) is a single required line with per data column a name for that column.
The order of the columns does not matter but the order of the names should match the order of the value columns.
MDT automatically links protocol columns to the protocol parameters of a model (see :ref:`protocol_parameters`), so make sure that the columns names are identical to the
protocol parameter names in your model.

The pre-provided list of column names is:

* **b**, the b-values in :math:`s/m^2` (:math:`b = \gamma^2 G^2 \delta^2 (\Delta-\delta/3)` with :math:`\gamma = 2.675987E8 \: rads \cdot s^{-1} \cdot T^{-1}`)
* **gx, gy, gz**, the gradient direction as a unit vector
* **Delta**, the value for :math:`{\Delta}` in seconds
* **delta**, the value for :math:`{\delta}` in seconds
* **G**, the gradient amplitude
* **TE**, the echo time in seconds
* **TR**, the repetition time in seconds

Note that MDT expects the columns to be in **SI units**.

The protocol dependencies change per model and MDT issues a warning if a required column is missing from the protocol.
If no b-value is provided, MDT will calculate one using Delta, delta and G.
If the b-value and two out of three of ``{Delta, delta, G}`` are given, the provided b-value will take preference.
If at least three of ``{b, Delta, delta, G}`` are given the missing value will be calculated automatically when required.

A protocol can be created from a bvec/bval pair of files using the command line, python shell or GUI.
Please see the relevant sections in :ref:`analysis` for more details on creating a protocol.


.. _concepts_problem_data_models:

************
Problem data
************
In MDT, all data needed to fit a model is stored independently from the model in a :py:class:`~mdt.utils.DMRIProblemData` object.
An instance of this object needs to be created before fitting a model.
Then, during model fitting, the model loads the relevant data for the computations.

The easiest way to instantiate a problem data object is by using the function :func:`~mdt.utils.load_problem_data`.
At a bare minimum, this function requires:

* ``volume_info``, the diffusion weighted volume, or MRI data in general in the case of other MRI modalities
* ``protocol``, an Protocol instance containing the protocol information
* ``mask``, the mask (3d) specifying which voxels to use for the computations

Additionally you can provide a *dictionary of static maps*, a *gradient deviations* file and a *standard deviation* for the noise.
For the standard deviation you have the choice to either provide a single value, an ndarray with one value per voxel or the string 'auto'.
If 'auto' is given MDT will use one or more of the :ref:`dynamic_modules_noise_std_estimators` to estimate the standard deviation of the
noise of the unweighted diffusion MRI in the complex plain.
The gradient deviations should be in the format described by the HCP Wu-Minn project.
The static maps then is a dictionary of maps with one value per voxel or, alternatively, one value per voxel per volume.


***************
Dynamic modules
***************
Extending and adapting MDT with new models is made easy using dynamically loadable modules placed in your home folder.
These modules are Python files placed in the ``.mdt`` folder in your home drive and are reloaded every time MDT is started.
Users are free to add, remove and modify components in this folder and MDT will pickup the changes after a restart.
See :ref:`dynamic_modules` for more information.


.. _concepts_cl_code:

*******
CL code
*******
The compartment models in MDT are programmed in the OpenCL C language (CL language from hereon).
See (https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/mathFunctions.html) for a quick reference on the available math functions in OpenCL.

When optimizing a multi-compartment model, MDT combines the CL code of all your compartments into one large function and uses MOT to optimize this function using the OpenCL framework.
See this figure for the general compilation flow in MDT:

.. image:: _static/figures/mdt_compilation_flow.png

There is one catch to this setup, one must avoid naming conflicts.
Since OpenCL kernels have a single global function namespace and a lot of functions are combined into one kernel
(e.g. compartment models, optimization routines, library routines etc.) it is possible to have naming conflicts.
If you follow the compartment modeling guidelines in :ref:`model_building_defining_compartments` you are generally fine.

To support both single and double floating point precision, MDT uses the ``mot_float_type`` instead of ``float`` and ``double`` for most of the variables and function definitions.
During optimization and sampling, ``mot_float_type`` is type-defined to be either a float or a double, depending on the desired precision.
Of course this does not limit you to use ``double`` and ``float`` as well in your code.
