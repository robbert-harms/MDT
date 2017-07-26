.. _mdt_example_data:

****************
MDT example data
****************
MDT comes pre-loaded with some example data that allows you to quickly get started using the software.
This example data can be obtained in the following ways:

* **GUI**: Open the model fitting GUI and find in the menu bar: "Help -> Get example data".
* **Command line**: Use the command :ref:`cli_index_mdt-get-example-data`:

.. code-block:: console

    $ mdt-get-example-data .


* **Python API**: Use the function :func:`mdt.utils.get_example_data`:

.. code-block:: python

    import mdt
    mdt.get_example_data('/tmp')


There are two MDT example datasets, a *b1k_b2k* dataset and a *multishell_b6k_max* dataset, both acquired in the same session on a Siemens Prisma system, on the VE11C software line,
with the standard product diffusion sequence at 2mm isotropic with GRAPPA in-plane acceleration factor 2 and 6/8 partial fourier (no multiband/simultaneous multi-slice).


The *b1k_b2k* has a shell of b=1000s/mm^2 and of b=2000s/mm^2 and is very well suited for e.g. Tensor, Ball&Stick and NODDI.
In this, the b=1000s/mm^2 shell is the standard Jones 30 direction table, including 6 b0 measurements at the start.
The b=2000s/mm^2 shell is a 60 whole-sphere direction set create with an electrostatic repulsion algorithm and has another 7 b0 measurements, 2 at the start of the shell and then one every 12 directions.


The *multishell_b6k_max* dataset has 6 b0's at the start and a range of 8 shells between b=750s/mm^2 and b=6000s/mm^2 (in steps of 750s/mm^2) with an increasing number of directions per shell
(see `De Santis et al., MRM, 2013 <http://dx.doi.org/10.1002/mrm.24717>`_) and is well suited for CHARMED analysis and other models that require high b-values (but no diffusion time variations).
