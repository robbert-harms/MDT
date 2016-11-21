**************
Batch profiles
**************
Batch profiles are part of the batch processing engine in MDT.
They specify how all the necessary data for model fitting (protocol, nifti volumes, brain mask, noise std, etc.) can be loaded from a directory containing one or more subjects.
This information is then used during for example batch fitting where MDT routinely asks the batch profile for the data of each of the subjects to be able to process them.

For example, suppose you have a directory containing a lot of subjects on which you want to run the NODDI analysis using MDT.
One way to do this would be to write a script that loops over the directories and calls the right fitting commands per subject.
Another approach would be to write a batch profile and use the batch processing utilities in MDT to process the datasets automatically.
The advantage of the latter is that there are multiple batch processing tools in MDT such as :func:`~mdt.batch_fit`, :func:`~mdt.batch_utils.collect_batch_fit_output` and
:func:`~mdt.batch_utils.run_function_on_batch_fit_output` that allow you to easy manipulate large groups of subjects.

Since the batch profiles form an specification of the directory structure, MDT can guess from a given directory which batch profile to use for that directory.
This makes batch processing in some instances as simple as using:

.. code-block:: bash

    $ mdt-batch-fit .


