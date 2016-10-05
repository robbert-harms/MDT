Installation
------------
.. highlight:: console

MDT uses the Maastricht Optimization Toolbox (MOT) for the computations. Please install MOT first (https://github.com/cbclab/MOT), afterwards this installation should be fairly straightforward.

Linux (Ubuntu)
""""""""""""""
It is possible to install every Python dependency with pip, but in general
native Ubuntu packages are preferred. To install most of the dependencies, please use:

``apt-get install python3-numpy python3-nibabel python3-pyqt5 python3-matplotlib python3-six python3-yaml python3-argcomplete``

Then, install MDT with pip (no Ubuntu package is available yet):

``pip3 install MDT``

On older Ubuntu systems (<15.10) some dependencies will be recompiled with pip. This might fail because of missing some packages, install with:

``apt-get install libpng-dev libfreetype6-dev libxft-dev``


Windows
"""""""
Open an Anaconda shell and use:

``pip install MDT``
