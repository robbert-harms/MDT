##############
Advanced usage
##############

.. _batch_fitting:

*************
Batch fitting
*************
MDT has support for batch fitting, allowing you to analyze many subjects with just one command.
This feature uses :ref:`dynamic_modules_batch_profiles` to gather information about the subjects in a given directory and apply the desired operation on each of those subjects.

When no batch profile is specified in the batch fitting command, MDT tests all available batch profiles and selects from that the best fitting profile automatically.
This means that for common directory layouts no batch profile need to be specified.

There are various batch profiles pre-provided in MDT, like profiles for the HCP-MGH and HCP Wu-Minn folder layouts, and simple layouts expecting one subject per directory.
Adding new profiles specific for your study is as simple as adding a new BatchProfile to the correct folder in the ``.mdt`` directory.
Alternatively, you could rename your folders and data to match the ``DirPerSubject`` profile.

The examples in this section use the same two datasets as described in the single subject analysis section above.

Using the command line
======================
For example, to run ``BallStick_r1`` on the two provided example datasets you can use the command :ref:`cli_index_mdt-batch-fit`:

.. code-block:: console

    $ cd mdt_example_data
    $ mdt-batch-fit . 'BallStick_r1 (Cascade)'


Or, another example, if you want to analyze ``NODDI`` on all your downloaded HCP datasets, you can use:

.. code-block:: console

    $ mdt-batch-fit ./download_dir/ 'NODDI (Cascade)'

and it will autodetect both the MGH and Wu-Minn layout and fit NODDI to all the subjects.


Using Python
============
To run ``BallStick_r1`` on the two provided example datasets using Python you can use the command :func:`~mdt.batch_fit`.
For example:

.. code-block:: python

    mdt.batch_fit('mdt_example_data', ['BallStick_r1 (Cascade)'])


Or, another example, if you want to analyze ``NODDI`` on all your downloaded HCP datasets, you can use:

.. code-block:: python

    mdt.batch_fit('~/download_dir/', ['NODDI (Cascade)'])


and it will autodetect the MGH and WuMinn layout and fit NODDI to all the subjects.


.. _advanced_usage_visualization_gui:

*****************
Visualization GUI
*****************
As explained in the getting started guide (:ref:`view_maps_gui`), the visualization GUI is a small convenience utility for visualizing nifti files.
This chapter goes a little bit more in depth in the functionality of this GUI.


GUI Layout
==========
The main body of the GUI is split into two parts, the control panel on the left and the display panel on the right.
All plot configuration options can be updated using the control panel, which includes things like colormaps, zooming, font sizes, rotations, clipping and more.
Changes in the plot settings are automatically reflected in the display panel.
This panel shows the current plot using matplotlib as a plotting backend.

On the bottom of the control panel are a few switches to control the rendering of the plots in the display panel.
The checkbox "Auto render" disables/enables the automatic rerendering of the plots, allowing you to change multiple settings without rerendering delays.
A manual redraw can be forced using the "Redraw" button.
The back and forward arrows allow you to undo or redo updates to the plot settings.

When hovering a map, the bottom right of the window shows some basic voxel statistics for the current mouse position.
The first tuple (``(63, 50)`` in the example screenshot below), shows the position of the hovered voxel in the current viewport.
The second tuple (``(63, 50, 0)`` in the example) shows the absolute position of the hovered voxel inside the nifti file (the two tuples can be different when the map is zoomed in or rotated).
Finally, the last item shows the value/intensity of the hovered voxel.

.. figure:: _static/figures/mdt_maps_visualizer_intro.png

    Screenshot of the GUI running in Linux


Control panel
-------------
The control panel consists of three tabs, "General", "Maps" and "Textual".
The first tab contains general options that all apply to the figure in total.
For instance, the zoom settings allows you to zoom in on all maps at the same time and the rotate option under miscellaneous rotates all displayed figures.

The second tab is for map specific options, here one can set plot configuration options that apply only to a single map.
After having selected the map you wish to change using the drop down box on the top of the panel you can then update all the values in the tab and the changes will be applied to the chosen map.
A common thing to change is the "Scale" of the map, which sets the range of the colormap to the defined scale, left empty this will auto-select a good scaling.
Another thing that can be set is the "Clipping" which will actively clip the data to be within the defined range.

The last tab of the control panel contains a live text area that allows you to change all plot settings (the general and the map specific) using a text editor.
This text box is automatically updated whenever one of the settings on the other tabs changes and vice versa.
This text box can be used to, for example, copy paste a configuration from one plot into another to let both reflect the exact same settings.
For more information on this feature, please see the next section, :ref:`maps_gui_plot_configuration`.

.. figure:: _static/figures/mdt_maps_visualizer_control_panel.png

    Figure showing the three tabs of the control panel combined into one figure, with the general tab on the left,
    the map specific options in the center and the textual input tab on the right.


.. _maps_gui_plot_configuration:


Plot configuration
==================
Any instance of the visualization routine consists of two things, data and a plot configuration.
The data is commonly loaded by selecting a directory with maps to load (or, using the Python API, a dictionary with maps).
Then, the selected maps or a subset of the maps, are visualized according to the plot configuration.
This plot configuration can be configured implicitly by using the "General" and "Maps" tag or explicitly using the "Textual" tab.

The plot configuration is commonly stored as a YAML formatted string that lists the various options as dictionary elements.
For example, the following configuration is a configuration for BallStick_r1 model fitting results where we set the zoom and the plot titles using the control panels.
As an example, after having followed the analysis getting started guide with the BallStick_r1 model, you could try to copy paste this example configuration in the "Textual" tab in the viewer.
It should then update the plot to reflect this configuration.

.. code-block:: yaml

    colorbar_nmr_ticks: 4
    colormap: hot
    dimension: 2
    flipud: false
    font: {family: sans-serif, size: 14}
    grid_layout:
    - Rectangular
    - cols: null
      rows: null
      spacings: {bottom: 0.04, hspace: 0.2, left: 0.1, right: 0.86, top: 0.97, wspace: 0.5}
    interpolation: bilinear
    map_plot_options:
      w_ball.w:
        clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
        colorbar_label: null
        colormap: null
        mask_name: null
        scale: {use_max: true, use_min: true, vmax: 1.0, vmin: 0.0}
        show_colorbar: true
        title: Isotropic (w_ball.w)
        title_spacing: null
      w_stick.w:
        clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
        colorbar_label: null
        colormap: null
        mask_name: null
        scale: {use_max: true, use_min: true, vmax: 1.0, vmin: 0.0}
        show_colorbar: true
        title: Anisotropic (w_stick.w)
        title_spacing: null
    maps_to_show: [w_ball.w, w_stick.w]
    mask_name: null
    rotate: 90
    show_axis: false
    show_colorbar: true
    slice_index: 0
    title: null
    volume_index: 0
    zoom:
      p0: {x: 18, y: 4}
      p1: {x: 85, y: 98}


An alternative way of saving this configuration file is by using the "Export settings" and "Import settings" in the menu.
This will provide easy ways of loading and saving the configuration file as a ``.conf`` file in YAML format.

Storing the plot configuration in a file has the additional advantage that you can automatize figure generation.
For example, in Python, one could use the function :func:`mdt.write_view_maps_figure`:

.. code-block:: python

    import mdt

    with open('plot_config.conf', 'r') as f:
        config = f.read()

    mdt.write_view_maps_figure(
        './b1k_b2k_example_slices_24_38_mask/BallStick_r1',
        '/tmp/test.png',
        config=config,
        width=1280,
        height=720,
        dpi=100)


to export the maps using the provided configuration as an ``.png`` image.
From the command line one could use, for a similar effect the command :ref:`cli_index_mdt-view-maps`:

.. code-block:: console

    $ mdt-view-maps \
         "./b1k_b2k_example_slices_24_38_mask/BallStick_r1" \
         --config "./plot_config.conf" \
         --to-file "/tmp/test.png" \
         --width 1280 \
         --height 720 \
         --dpi 100


Please note that example scripts like this are automatically created when you save a plot as an image using the GUI.

