.. _view_maps_gui:

*******************
MDT maps visualizer
*******************
The MDT maps visualizer is a small convenience utility to visually inspect multiple nifti files simultaneously.
In particular, it is useful for quickly visualizing model fitting results.

This viewer is by far not as sophisticated as for example ``fslview`` and ``itksnap``, but that is also not its intention.
The primary goal of this visualizer is to quickly display model fitting results to evaluate the quality of fit.
A side-goal of the viewer is the ability to create reproducible and paper ready figures showing slices of various volumetric maps.

Features include:

* output figures as images (``.png`` and ``jpg``) and as vector files (``.svg``)
* the ability to store plot configuration files that can later be loaded to reproduce figures
* easily display multiple maps simultaneously

Some usage tip and tricks are:

* Click on a point on a map for an annotation box, click outside a map to disable
* Zoom in by scrolling in a plot
* Move the zoom box by clicking and dragging in a plot
* Add new nifti files by dragging them from a folder into the GUI


For more details about the visualization GUI, please see :ref:`advanced_usage_visualization_gui` in advanced usage.

The following is a screenshot of the GUI displaying NODDI results of the b1k_b2k MDT example data slices.

.. figure:: _static/figures/mdt_view_maps_gui.png

    The MDT map visualizer in Linux


The MDT maps visualizer can be started in three ways, from the MDT analysis GUI, from the command line and using the Python API.
With the command line, the visualizer can be started using the command :ref:`cli_index_mdt-view-maps`, for example:

.. code-block:: console

    $ mdt-view-maps .

In Windows, one can also type this command in the start menu search bar to load and start the GUI.
Using Python, the GUI can be started using the command :func:`mdt.view_maps`, for example:

.. code-block:: python

    >>> import mdt
    >>> mdt.view_maps('output/NODDI')


Finally, using the MDT analyis GUI, the maps visualizer can be started using the button on the last tab:


.. figure:: _static/figures/mdt_start_visualizer_from_gui.png

    Starting the maps visualizer from the analysis GUI.
