Installation
************
.. highlight:: console

MDT uses the Maastricht Optimization Toolbox (MOT) for the computations. Please install MOT first (https://github.com/cbclab/MOT), afterwards this installation should be fairly straightforward.

Linux (Ubuntu)
==============
After having installed MOT, we can install MDT. It is possible to install every Python dependency with pip, but in general
native Ubuntu packages are preferred. To install most of the dependencies, please use:

.. code-block:: bash

    $ apt-get install python3-numpy python3-nibabel python3-pyqt5 python3-matplotlib python3-six python3-yaml python3-argcomplete


Then, install MDT with pip (no Ubuntu package is available yet):

.. code-block:: bash

    $ pip3 install MDT

On older Ubuntu systems (<15.10) some dependencies will be recompiled with pip. This might fail because of missing some packages, install these with:

.. code-block:: bash

    $ apt-get install libpng-dev libfreetype6-dev libxft-dev

and then repeat the pip3 installation.


Windows
=======
After having installed MOT, we can install MDT. Open an Anaconda shell and use:

.. code-block:: bash

    $ pip install MDT


Initialization
==============
After installation we need to initialize the MDT components folder in your home folder. Use:

.. code-block:: bash

    $ mdt-init-user-settings

in your bash or Anaconda shell to install the dynamically loadable modules.
Please see the section :ref:`concepts_dynamical_modules` for more information on these modules.


Test the installation
=====================
If all went well and MDT is installed and initialized, please


.. code-block:: bash

    $ mdt-init-user-settings

in your bash or Anaconda shell to install the dynamically loadable modules.
Please see the section :ref:`concepts_dynamical_modules` for more information on these modules.

