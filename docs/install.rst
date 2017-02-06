############
Installation
############


*********************
Ubuntu / Debian Linux
*********************
Using the package manager, installation in Ubuntu and Debian is relatively straightforward.

For **Ubuntu >= 16** the MOT package can be installed with a Personal Package Archive (PPA) using:

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
Please install MOT first (https://mot.readthedocs.io/en/latest/install.html#windows) and afterwards this installation should be fairly straightforward.

Note that MDT depends on PyQt5 so make sure you do not attempt to run it in an environment with PyQt4 or earlier.
If you followed the MOT install guide and installed the Anaconda version 4.2 or higher with Python3.x, you should be fine.
Again, see https://mot.readthedocs.io/en/latest/install.html#windows for details.

Having followed the MOT install guide we can now install MDT.
Open an Anaconda console and use:

.. code-block:: bash

    $ pip install mdt

If that went well please continue with the `Initialization`_ below.


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
Type in your console:

.. code-block:: bash

    $ mdt-gui

or, equivalently,

.. code-block:: bash

    $ MDT

to check if the GUI works.
If this fails, double check the above installation steps.

Another command to try is:

.. code-block:: bash

    $ mdt-list-devices

This should print a list of OpenCL devices in your computer.
If this returns nothing you may be lacking OpenCL drivers for your machine.
Please refer to the section :ref:`faq_no_opencl_device_found` for help on this problem.
