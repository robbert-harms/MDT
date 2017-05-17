############
Installation
############


*********************
Ubuntu / Debian Linux
*********************
Using the package manager, installation in Ubuntu and Debian is relatively straightforward.

For **Ubuntu >= 16** the MOT package can be installed from our Personal Package Archive (PPA) using:

.. code-block:: bash

    $ sudo add-apt-repository ppa:robbert-harms/cbclab
    $ sudo apt-get update
    $ sudo apt-get install python3-mdt

By using a PPA your Ubuntu system can update MDT automatically whenever a new version is out.

For **Debian**, and **Ubuntu < 16**, using a PPA is not possible (because of missing dependent packages) and we need a more manual installation.
Please install the dependencies first:

.. code-block:: bash

    $ sudo apt install python3 python3-pip python3-pyopencl \
        python3-numpy python3-nibabel python3-pyqt5 \
        python3-matplotlib python3-six python3-yaml \
        python3-argcomplete libpng-dev libfreetype6-dev libxft-dev


Note that ``python3-nibabel`` may need NeuroDebian to be available on your machine.
An alternative is to use ``pip3 install nibabel`` instead.

Next, install MDT with:

.. code-block:: bash

    $ sudo pip3 install mdt

This might recompile a few packages to use the latest versions.

After installation please continue with the section `Initialization`_ below.


*******
Windows
*******
MDT uses the Maastricht Optimization Toolbox (MOT) for all analysis computations.
Please install MOT first (https://mot.readthedocs.io/en/latest/install.html#windows). Afterwards this installation should be fairly straightforward.

Note that MDT depends on PyQt5 so make sure you do not attempt to run it in an environment with PyQt4 or earlier.
If you followed the MOT install guide and installed the Anaconda version 4.2 or higher with Python3.x, you should be fine.
Again, see https://mot.readthedocs.io/en/latest/install.html#windows for details.

Having followed the MOT install guide we can now install MDT.
Open an Anaconda console and use:

.. code-block:: bash

    $ pip install mdt

If that went well please continue with the `Initialization`_ below.


***
Mac
***
Installation on Mac is pretty easy using the Anaconda 4.2 or higher Python distribution.
Please download and install the Python3.x 64 bit distribution, version 4.2 or higher which includes PyQt5,
from `Anaconda <https://www.continuum.io/downloads>`_ and install it with the default settings.

Afterwards, open a terminal and type:

.. code-block:: bash

    $ pip install mdt


To install MDT to your system.
If that went well please continue with the `Initialization`_ below.

Please note that Mac support is experimental due to the unstable nature of the OpenCL drivers in Mac.
Users running Running MDT with the GPU as selected device may experience crashes.
Running MDT in the CPU seems to work though.


**************
Initialization
**************
After installation we need to initialize the MDT components folder in your home folder. Use:

.. code-block:: bash

    $ mdt-init-user-settings

in your bash or Anaconda console to install the MDT model library to your home folder.


*********************
Test the installation
*********************
If all went well and MDT is installed and initialized, we can now perform some basic tests to see if everything works well.
The first command to try is:

.. code-block:: bash

    $ mdt-list-devices

which should print to the console a list of available CL devices.
If this crashes or if there are no devices returned, please check to see if your OpenCL drivers are correctly installed.
If this crashes with an exception then most likely the OpenCL environment can not be found, see :ref:`faq_clGetPlatformIDs_failed`.
If it works but no devices can be found then please refer to the section :ref:`faq_no_opencl_device_found`.

Next, one could try starting the graphical interface using:

.. code-block:: bash

    $ mdt-gui

or, equivalently,

.. code-block:: bash

    $ MDT

This should start the GUI. If there are problems in this stage it is most likely related to Qt problems.
Please check if you have installed the Qt5 package and not the Qt4 package.
