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



********
Analysis
********

Why does the noise standard deviation differ when using another mask?
=====================================================================
By default MDT tries to estimate the noise standard deviation of the images in the complex domain.
This standard deviation is used in the analysis as the standard deviation in the likelihood function (commonly Offset-Gaussian).
This standard deviation is commonly estimated using an average of per-voxel estimations.
When a different mask is used there are different voxels used for the standard deviation estimation and hence the resulting value differs.

To prevent this from happening it is suggested that researchers estimate the noise std. beforehand with a whole brain mask and use the obtained std. in all other analysis.


.. only:: html

    .. rubric:: References

.. bibliography:: references.bib
    :style: plain
    :filter: {"faq"} & docnames
