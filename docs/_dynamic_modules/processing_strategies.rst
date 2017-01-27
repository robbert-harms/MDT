*********************
Processing strategies
*********************
Processing strategies are used by the model optimization and model sampling routines to split the voxels in batches which allows MDT to save intermediate results.
Regular end users will find no need to add strategies, but may still be interested in using a specific strategy for a specific model.
This can be set using the configuration setting ``processing_strategies``.

The strategies work by returning an iterator that yields batches of voxels to optimize or sample.
A common strategy is the ``ProtocolDependent`` strategy which changes the batch size depending on the length of the protocol in use.
A larger protocol file requires more memory in processing which slows down the overall computations.
In this case, decreasing the amount of voxels sometimes increases performance.

More in general, the optimum batch size is the one in which the memory of the GPU is completely saturated with voxels to process and nothing more.
Setting the batch size too low and compute power is wasted on the GPU (not so much on the CPU).
Setting the batch size too high results in less intermediate storage and long kernel execution times which may temporarily freeze your desktop.

For models that come pre-supplied with MDT, care has been taken to find an optimal set of processing strategies that work with most operating systems and most GPU's/CPU's.
