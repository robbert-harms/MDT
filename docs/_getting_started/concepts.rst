Concepts
========

This section introduces the user to a few concepts the Maastricht Diffusion Toolbox introduces in the analysis workflow.

.. _concepts_protocol:

Protocol
--------
In MDT, the Protocol contains the settings used for the MRI measurements and is by convention stored in a protocol file
with suffix ``.prtcl``. Such a protocol file is a tab delimited values file containing for every volume (rows) and for every setting (columns)
a single value..

The following is an example of a simple MDT protocol file::

    #gx,gy,gz,b
    0.000e+00    0.000e+00    0.000e+00    0.000e+00
    5.572e-01    6.731e-01    -4.860e-01   1.000e+09
    4.110e-01    -5.254e-01   -7.449e-01   1.000e+09
    ...


Or, for a more advanced protocol file::

    #gx,gy,gz,Delta,delta,TE,b
    -0.000e+00  0.000e+00   0.000e+00   2.179e-02   1.290e-02   5.700e-02   0.000e+00
    2.920e-01   1.7100e-01  -9.409e-01  2.179e-02   1.290e-02   5.700e-02   3.000e+09
    -9.871e-01  -8.538e-03  -1.595e-01  2.179e-02   1.290e-02   5.700e-02   5.000e+09
    ...


The header (starting with #) is a single required line with per column the setting name (the order does not matter). These setting names should match the names of the protocol parameters of the
desired model (see :ref:`protocol_parameters`). All other lines should contain tab delimited values and are interpreted as rows, with for every volume in the MRI dataset
a matching row. This automatically enables *multi-shell protocol* files. Note that MDT expects the columns to be in **SI units**.

The pre-provided list of setting / column names is:

* **b**, the b-values in :math:`s/m^2` (:math:`b = \gamma^2 G^2 \delta^2 (\Delta−\delta/3)` with :math:`\gamma = 2.675987E8 \: rads \cdot s^{-1} \cdot T^{-1}`)
* **gx, gy, gz**, the gradient direction as a unit vector
* **Delta**, the value for :math:`{\Delta}` in seconds
* **delta**, the value for :math:`{\delta}` in seconds
* **G**, the gradient amplitude
* **TE**, the echo time in seconds
* **TR**, the repetition time in seconds

Not all models need all of these settings and MDT will issue warnings if a required setting is missing. If no b-value is provided MDT
will calculate one using Delta, delta and G. If the b-value and two out of three of ``{Delta, delta, G}`` are given, the provided b-value will take preference.
If at least three of ``{b, Delta, delta, G}`` are given the other value will be calculated automatically when required.

One can dynamically add column names to MDT as shown in :ref:`model_building_protocol_parameters`. A protocol can be created from a bvec/bval pair of files using the
command line, python shell or GUI. Please see the relevant sections in :ref:`analysis` for more details on creating a protocol.


.. _concepts_problem_data_models:

Problem data and models
-----------------------
TODO

* DMRIProblemData class
* interaction between models and problem data


Dynamic modules
---------------
MDT can easily be adapted and extended with new models using dynamically loadable modules. These modules are Python files
placed in the ``.mdt`` folder in your home drive and are reloaded every time MDT is started.
Users are free to add, remove and modify these components. See :ref:`dynamic_modules` for more information.
