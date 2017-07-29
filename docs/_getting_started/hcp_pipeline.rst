************
HCP Pipeline
************
MDT comes pre-installed with Human Connectome Project (HCP) compatible pipelines for the MGH and the WuMinn 3T studies.
To run, please change directory to where you downloaded your (pre-processed) HCP data (MGH or WuMinn) and execute:

.. code-block:: console

    $ mdt-batch-fit . 'NODDI (Cascade)'

and it will autodetect the study in use and fit your selected model to all the subjects.

Some of the models you can use are: ``NODDI``, ``ActiveAx``, ``CHARMED_r1``, ``Tensor``, ``BallStick_r1`` and cascaded versions of these.
For a complete list of models run the command :ref:`cli_index_mdt-list-models` and for more information about cascades see :ref:`concepts_composite_and_cascade_models`.
