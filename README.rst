============================
Maastricht Diffusion Toolkit
============================

.. image:: https://badge.fury.io/py/mdt.png
    :target: http://badge.fury.io/py/mdt

A diffusion toolkit for parallelized sampling and optimization of diffusion data.

* Free software: LGPL v3 license
* Full documentation: https://mdt.readthedocs.org
* Project home: https://github.com/robbert-harms/MDT
* Uses the `GitLab workflow <https://docs.gitlab.com/ee/workflow/gitlab_flow.html>`_
* Tags: diffusion, dMRI, MRI, optimization, parallel, opencl, python


Installation
------------
.. highlight:: console

MDT uses the Maastricht Optimization Toolbox (MOT) for the computations. Please install MOT first (https://github.com/robbert-harms/MDT), afterwards this installation should be fairly straightforward.


|
Installing MDT
^^^^^^^^^^^^^^
With MOT installed, you can now install MDT.

Linux (Ubuntu)
""""""""""""""
It is possible to install every Python dependency with pip, but in general
native Ubuntu packages are prefered. To install most of the dependencies, please use:

``apt-get install python3-numpy python3-nibabel python3-pyqt5 python3-matplotlib python3-six python3-yaml python3-argcomplete``

Then, install MDT with pip (no Ubuntu package is available yet):

``pip3 install MDT``

On older Ubuntu systems (<15.10) some dependencies will be recompiled with pip. This might fail because of missing some packages, install with:

``apt-get install libpng-dev libfreetype6-dev libxft-dev``


Windows
"""""""
Open an Anaconda shell and use:

``pip install MDT``
