############
Installation
############


*********************
Ubuntu / Debian Linux
*********************
Using the package manager, installation in Ubuntu and Debian is easy.

For **Ubuntu >= 16** the MOT package can be installed with a Personal Package Archive (PPA) using:

.. code-block:: bash

    $ sudo add-apt-repository ppa:robbert-harms/cbclab
    $ sudo apt-get update
    $ sudo apt-get install python3-mdt

Using such a PPA ensures that your Ubuntu system can update the MDT package automatically whenever a new version is out.
For **Debian**, and **Ubuntu < 16**, using a PPA is not possible and we need a more manual installation.
Please install the dependencies first:

.. code-block:: bash

    $ sudo apt install python3 python3-pip python3-pyopencl \
        python3-numpy python3-nibabel python3-pyqt5 \
        python3-matplotlib python3-six python3-yaml \
        python3-argcomplete libpng-dev libfreetype6-dev libxft-dev

and then install MOT with:

.. code-block:: bash

    $ sudo pip3 install mdt

This might recompile a few packages to use the latest versions.
After installation please continue with the section `Initialization`_ below.

*******
Windows
*******
MDT uses the Maastricht Optimization Toolbox (MOT) for all modeling computations.
Please install MOT first (https://mot.readthedocs.io/en/latest/install.html#windows), afterwards this installation should be fairly straightforward.

Note that MDT depends on PyQt5 so make sure you do not attempt to run it in an environment with PyQt4 or earlier.
If you followed the MOT install guide and installed the Anaconda Python3.x 64 bit version 4.2 or higher, you should be fine.
See https://mot.readthedocs.io/en/latest/install.html#windows.

Having followed the MOT install guide we can now install MDT. Open an Anaconda console and use:

.. code-block:: bash

    $ pip install mdt

If that went well please continue with the `Initialization`_ below.


**************
Initialization
**************
After installation we need to initialize the MDT components folder in your home folder. Use:

.. code-block:: bash

    $ mdt-init-user-settings

in your bash or Anaconda console to install the MDT library of models to your home folder.


*********************
Test the installation
*********************
If all went well and MDT is installed and initialized, we can now perform some basic tests to see if all works well.
Type in your console:

.. code-block:: bash

    $ mdt-gui

or, equivalently,

.. code-block:: bash

    $ MDT

to check if the GUI works. If this fails, double check the above installation steps.

Another command to try is:

.. code-block:: bash

    $ mdt-list-devices

This should print a list of CL enabled devices in your computer.
If this returns nothing you may be lacking OpenCL drivers for your machine.
Please refer to the section :ref:`faq_no_opencl_device_found` for help on this problem.
