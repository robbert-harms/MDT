.. _batch_fitting:

*************
Batch fitting
*************
MDT features a batch fitting routine that can analyze many subjects with just one command.
This feature uses :ref:`dynamic_modules_batch_profiles` to gather information about the subjects in a given directory and apply the desired operation on each of those subjects.

When no batch profile is specified in the batch fitting command, MDT tests all available batch profiles and selects from that the best fitting profile automatically.
This means that for common directory layouts no batch profile need to be specified.

There are various batch profiles pre-provided in MDT, like profiles for the HCP-MGH and HCP Wu-Minn folder layouts, and simple layouts expecting one subject per directory.
Adding new profiles specific for your study is as simple as adding a new BatchProfile to the correct folder in the ``.mdt`` directory.
Alternatively, you could rename your folders and data to match the ``DirPerSubject`` profile.

The examples in this section use the same two datasets as described in the single subject analysis section above.

Using the command line
======================
For example, to run ``BallStick_r1`` on the two provided example datasets you can use the command :ref:`cli_index_mdt-batch-fit`:

.. code-block:: console

    $ cd mdt_example_data
    $ mdt-batch-fit . 'BallStick_r1 (Cascade)'


Or, another example, if you want to analyze ``NODDI`` on all your downloaded HCP Wu-Minn datasets, you can use:

.. code-block:: console

    $ mdt-batch-fit ~/download_dir/ 'NODDI (Cascade)'

and it will autodetect the Wu-Minn layout and fit NODDI to all the subjects.


Using Python
============
To run ``BallStick_r1`` on the two provided example datasets using Python you can use the command :func:`~mdt.batch_fit`.
For example:

.. code-block:: python

    mdt.batch_fit('mdt_example_data', ['BallStick_r1 (Cascade)'])


Or, another example, if you want to analyze ``NODDI`` on all your downloaded HCP Wu-Minn datasets, you can use:

.. code-block:: python

    mdt.batch_fit('~/download_dir/', ['NODDI (Cascade)'])


and it will autodetect the Wu-Minn layout and fit NODDI to all the subjects.
