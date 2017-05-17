############################
Maastricht Diffusion Toolbox
############################
The Maastricht Diffusion Toolbox, MDT, is a framework and library for parallelized (GPU and multi-core CPU) diffusion Magnetic Resonance Imaging (MRI) modeling.
MDT's object oriented and modular design allows arbitrary user specification and combination of biophysical MRI compartment models, diffusion- and T1, T2, T2* based microstructure models,
likelihood functions and optimization algorithms.
MDT was designed with compatibility in mind and adheres to input, output and variable naming conventions used by other related software tools.
Many diffusion and relaxometry microstructure models are included, and new models can be added simply by adding Python script files.
MDT can be extended to other modalities and other parametric models estimated from data volumes varying along controlled parameters (such as b-values, diffusion times, TE, TM, flip angle, etc).
The parallelized accelerated computations allow for tens to hundred times faster model fitting, even on standard GPU (and/or CPU) hardware, making MDT ideal for large group studies or population studies.


*******
Summary
*******
* Free Open Source Software: LGPL v3 license
* Python and OpenCL based
* GUI, command line and python interface
* Scriptable modeling: write new compartment equations and combine compartments into models
* Full documentation: http://maastrichtdiffusiontoolbox.readthedocs.io
* Project home: https://github.com/cbclab/MDT
* Uses the `GitLab workflow <https://docs.gitlab.com/ee/workflow/gitlab_flow.html>`_
* Tags: diffusion, dMRI, MRI, optimization, parallel, opencl, python


************************
Quick installation guide
************************
The basic requirements for MDT are:

* Python 3.x (recommended) or Python 2.7
* OpenCL 1.2 (or higher) support in GPU driver or CPU runtime


**Linux**

For Ubuntu >= 16 you can use:

* ``sudo add-apt-repository ppa:robbert-harms/cbclab``
* ``sudo apt-get update``
* ``sudo apt-get install python3-mdt``


For Debian users and Ubuntu < 16 users, install MDT with:

* ``sudo apt-get install python3 python3-pip python3-pyopencl python3-numpy python3-nibabel python3-pyqt5 python3-matplotlib python3-six python3-yaml python3-argcomplete libpng-dev libfreetype6-dev libxft-dev``
* ``sudo pip3 install mdt``

Note that ``python3-nibabel`` may need NeuroDebian to be available on your machine. An alternative is to use ``pip3 install nibabel`` instead.


**Windows**

The installation on Windows is a little bit more complex and the following is only a quick reference guide.
For complete instructions please view the `complete documentation <https://maastrichtdiffusiontoolbox.readthedocs.org>`_.

* Install Anaconda Python 3.5
* Install MOT using the guide at https://mot.readthedocs.io
* Open an Anaconda shell and type: ``pip install mdt``


**Mac**

* Install Anaconda Python 3.5
* Open a terminal and type: ``pip install mdt``

Please note that Mac support is experimental due to the unstable nature of the OpenCL drivers in Mac, that is, users running MDT with the GPU as selected device may experience crashes.
Running MDT in the CPU seems to work though.


For more information and full installation instructions see https://maastrichtdiffusiontoolbox.readthedocs.org
