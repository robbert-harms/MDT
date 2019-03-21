################################
Microstructure Diffusion Toolbox
################################
The Microstructure Diffusion Toolbox (MDT) is a framework and library for microstructure modeling of magnetic resonance imaging (MRI) data.
The aim of MDT is to provide reproducible and comparable model fitting for MRI microstructure analysis.
As such, we provide a common platform for microstructure modeling including many models that can all be processed using the same optimization routines.
For maximum performance all models and algorithms were implemented to make use of all parallel processing capabilities of modern computers.
MDT combines flexible modeling with fast processing, targeting both model developers and data analysts.


*******
Summary
*******
* GPU accelerated processing
* Human Connectome Project (HCP) pipelines
* Includes CHARMED, NODDI, BinghamNODDI, NODDIDA, NODDI-DTI, ActiveAx, AxCaliber, Ball&Sticks, Ball&Rackets, Kurtosis, Tensor, VERDICT, qMT, and relaxometry (T1, T2) models.
* Includes Gaussian, Offset-Gaussian and Rician likelihood models
* Includes Powell, Levenberg-Marquardt and Nelder-Mead Simplex optimization routines
* Includes multiple (adaptive) MCMC sampling algorithms
* Supports hyperpriors on parameters
* Supports gradient deviations per voxel and per voxel per volume
* Supports volume weighted objective function
* Supports adding your own models
* Offers Graphical, command line and python interfaces
* Computations are parallelized over voxels and over volumes
* Python and OpenCL based
* Free Open Source Software: LGPL v3 license
* Runs on Windows, Mac and Linux operating systems
* Runs on Intel, Nvidia and AMD GPU's and CPU's.


*****
Links
*****
* Full documentation: http://mdt_toolbox.readthedocs.io
* Project home: https://github.com/cbclab/MDT


************
HCP Pipeline
************
MDT comes pre-installed with Human Connectome Project (HCP) compatible pipelines for the MGH and the WuMinn 3T studies.
To run, after installing MDT, go to the folder where you downloaded your (pre-processed) HCP data (MGH or WuMinn) and execute:

.. code-block:: console

    $ mdt-batch-fit . NODDI

and it will autodetect the study in use and fit your selected model to all the subjects.


************************
Quick installation guide
************************
The basic requirements for MDT are:

* Python 3.x
* OpenCL 1.2 (or higher) support in GPU driver or CPU runtime


**Linux**

For Ubuntu >= 16 you can use:

* ``sudo add-apt-repository ppa:robbert-harms/cbclab``
* ``sudo apt-get update``
* ``sudo apt-get install python3-mdt python3-pip``
* ``sudo pip3 install tatsu``

For Debian users and Ubuntu < 16 users, install MDT with:

* ``sudo apt-get install python3 python3-pip python3-pyopencl python3-numpy python3-nibabel python3-pyqt5 python3-matplotlib python3-yaml python3-argcomplete libpng-dev libfreetype6-dev libxft-dev``
* ``sudo pip3 install mdt``

Note that ``python3-nibabel`` may need NeuroDebian to be available on your machine. An alternative is to use ``pip3 install nibabel`` instead.

A Dockerfile and Singularity recipe are also provided for installation with Intel OpenCL drivers pre-loaded (e.g. for containerized deployment on a CPU cluster).
For example, to install using Docker use ``docker build -f containers/Dockerfile.intel .``.


**Windows**

The installation on Windows is a little bit more complex and the following is only a quick reference guide.
For complete instructions please view the `complete documentation <https://mdt_toolbox.readthedocs.org>`_.

* Install Anaconda Python 3.*
* Install MOT using the guide at https://mot.readthedocs.io
* Open an Anaconda shell and type: ``pip install mdt``


**Mac**

* Install Anaconda Python 3.*
* Open a terminal and type: ``pip install mdt``

Please note that Mac support is experimental due to the unstable nature of the OpenCL drivers in Mac, that is, users running MDT with the GPU as selected device may experience crashes.
Running MDT in the CPU seems to work though.


For more information and full installation instructions see https://mdt_toolbox.readthedocs.org
