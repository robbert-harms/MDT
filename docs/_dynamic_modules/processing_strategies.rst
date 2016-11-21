*********************
Processing strategies
*********************
Processing strategies determine the size of the processed batches and indirectly the intervals at which intermediate results are saved.
The strategies work by returning an iterator that yields batches of voxels to optimize or sample.
Regular end users will find no need to add strategies, but may still be interested in using a specific strategy for a specific model.
This can be set using the configuration setting ``processing_strategies``.

A common strategy is the ``ProtocolDependent`` strategy which changes the batch size (the number of voxels to process at once) depending on the length of the protocol in use.
A larger protocol file requires more memory on the GPU during processing which slows down the overall computations.
In this case, decreasing the amount of voxels sometimes increases performance.

More in general, the optimum batch size is the one in which the memory of the GPU is completely saturated with voxels to process and not more than that.
Setting the batch size too low and compute power is wasted on the GPU (not so much on the CPU).
Setting the batch size too high results in less intermediate storage and long kernel execution times which may temporarily freeze your desktop.

For models that come pre-supplied with MDT, care has been taken to find an optimal set of processing strategies that work with most operating systems and most GPU's/CPU's.
