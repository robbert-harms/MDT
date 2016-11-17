Batch profiles
==============
Batch profiles are part of the batch fitting processing engine in MDT and specify how all the necessary data for model fitting
(protocol, nifti volumes, brain mask, noise std, etc.) need to be loaded from a directory containing one or more subjects.
For example, suppose you have a directory containing a lot of subjects on which you want to run the NODDI analysis using MDT.
One way to do this would be to write a script that loops over the directories and calls the rights fitting commands subject by subject.
Another approach would be to write a batch profile and use the batch processing utilities in MDT to process the datasets automatically.
The advantage of this is that next to batch fitting there are multiple batch processing tools such as :func:`~mdt.batch_utils.collect_batch_fit_output` and
:func:`~mdt.batch_utils.run_function_on_batch_fit_output` that allows one to easy manipulate large groups of subjects.

Since the batch profiles form an specification of the directory structure, MDT can guess from a given directory which batch profile to use for that directory.
This makes batch processing in some instances as simple as using:

.. code-block:: bash

    $ mdt-batch-fit .
