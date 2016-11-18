############################
Maastricht Diffusion Toolbox
############################
The Maastricht Diffusion Toolbox, MDT, is a framework and library for GPU (graphics card) accelerated diffusion modeling.
MDT's object oriented and modular design allows arbitrary user specification and combination of dMRI compartment models, diffusion microstructure models,
likelihood functions and optimization algorithms.
Many diffusion microstructure models are included, and new models can be added simply by adding Python script files.
The GPU accelerated computations allow for ~60x faster model fitting; e.g. the 81 volume example NODDI dataset can be fitted whole brain in about 40 seconds,
which makes MDT ideal for population studies.
Additionally, MDT can be extended to other modalities and models such as quantitative MRI relaxometry.

*******
Summary
*******
* Free software: LGPL v3 license
* Scriptable modeling
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


**Windows**

The installation on Windows is a little bit more complex and the following is only a quick reference guide.
For complete instructions please view the `complete documentation <https://maastrichtdiffusiontoolbox.readthedocs.org>`_.

* Install Anaconda Python 3.5
* Install MOT using the guide at https://mot.readthedocs.io
* Open a Anaconda shell and type: ``pip install mdt``


For more information and installation instructions, please see: https://maastrichtdiffusiontoolbox.readthedocs.org
