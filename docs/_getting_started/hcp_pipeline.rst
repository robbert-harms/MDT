************
HCP Pipeline
************
MDT comes pre-installed with Human Connectome Project (HCP) compatible pipelines for the MGH and the WuMinn 3T studies.
To run, please change directory to where you downloaded your (pre-processed) HCP data (MGH or WuMinn) and execute:

.. code-block:: console

    $ mdt-batch-fit . 'NODDI (Cascade)'

and it will autodetect the study in use and fit your selected model to all the subjects.

For more information on how this works, please see :ref:`batch_fitting`.
