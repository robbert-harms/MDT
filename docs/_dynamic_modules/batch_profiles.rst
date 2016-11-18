**************
Batch profiles
**************
Batch profiles are part of the batch processing engine in MDT.
They specify how all the necessary data for model fitting (protocol, nifti volumes, brain mask, noise std, etc.) need to be loaded from a directory containing one or more subjects.
During batch fitting, the MDT batch fit routine asks the batch profile for the data of the each of the subjects in turn to do the processing.

For example, suppose one has a directory containing a lot of subjects on which one wants to run NODDI analysis with MDT.
One way to do this would be to write a script that loops over the directories and calls the right fitting commands per subject.
Another approach would be to write a batch profile and use the batch processing utilities in MDT to process the datasets automatically.
The advantage of the latter is that there are multiple batch processing tools in MDT such as :func:`~mdt.batch_fit`, :func:`~mdt.batch_utils.collect_batch_fit_output` and
:func:`~mdt.batch_utils.run_function_on_batch_fit_output` that allow one to easy manipulate large groups of subjects.

Since the batch profiles form an specification of the directory structure, MDT can guess from a given directory which batch profile to use for that directory.
This makes batch processing in some instances as simple as using:

.. code-block:: bash

    $ mdt-batch-fit .


