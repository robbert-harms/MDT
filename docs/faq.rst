##########################
Frequently Asked Questions
##########################

.. contents:: Table of Contents
   :local:
   :backlinks: none


*********************
Installation problems
*********************

.. _faq_no_opencl_device_found:

No OpenCL device found
======================
If there are no CL devices visible in the GUI and the shell command ``mdt-list-devices`` returns nothing, no computations can be done using MDT.
To continue using MDT, OpenCL enabled hardware and corresponding drivers must be installed in your system.

Check devices
-------------
First, make sure that the graphics card and/or CPU in your system is capable of OpenCL acceleration.
To do so, look up the device name in your computer and find its specifications on the internet.
The device must support OpenCL and at least OpenCL version 1.2.

Check drivers
-------------
If your preferred device supports OpenCL and it does not show in MDT, you may be missing the device drivers.

If you would like to run the computations on a GPU (graphics card), please install the correct drivers for that card.
If you would like to run the computations on the CPU, you have two possibilities.
The first is to install an AMD graphics card, their drivers come pre-supplied with OpenCL drivers for CPU's (for both Intel and AMD).
If you do not have a graphics card, or you have an NVidia card, you will have to install the `Intel OpenCL Drivers <https://software.intel.com/en-us/articles/opencl-drivers>`_ to your system.


The GUI fails to start
======================
You have successfully installed MDT, but starting the Graphical User Interface (GUI) fails.
Most of the time this is because PyQt5 is not installed on your machine.
If you are using Windows, please make sure you are using the latest Anaconda version (> 4.2.0), at least one that includes PyQt5.
If you are using Linux, please make sure you download the correct Python PyQt5 package for your machine.
That should have been done automatically when you used the Ubuntu PPA.


********
Log file
********

What do the "model protocol options" entries in the log file mean?
==================================================================
This message represent itself either as "Applying model protocol options, we will use a subset of the protocol and DWI." or as "No model protocol options to apply, using original protocol."
In both cases, this refers to the :ref:`dynamic_modules_composite_models_protocol_options` set in the composite model you fitted or sampled.
In the first case we are applying model protocol options and hence we are only using a subset of all volumes, in the second case we use all volumes.
See the part about the :ref:`dynamic_modules_composite_models_protocol_options` for more information and on how to define these protocol options.
