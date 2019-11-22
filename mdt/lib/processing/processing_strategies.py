"""Contains the processing strategies (and workers) that define how to process a model (fitting and sample).

Globally, this module consists out of two public players, the :class:`ModelProcessingStrategy` and the
:class:`ModelProcessor`. The latter, the model processor contains information on how the model needs to be processed.
For example, optimization and sample are two different operations and hence require their own processing
implementation. Given that the model processor defines how to process the models, the model processing strategy
encapsulates how to process the processors. For example, a strategy may be to split all voxels into batches and optimize
those while saving intermediate results.
"""
import glob
import logging
import os
import shutil
import timeit
from contextlib import contextmanager
import numpy as np
import time
import gc
from numpy.lib.format import open_memmap
from mdt.lib.nifti import write_all_as_nifti
from mdt.utils import create_roi

__author__ = 'Robbert Harms'
__date__ = "2016-07-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelProcessingStrategy:
    """Model processing strategies define in how many parts a composite model is processed."""

    def process(self, processor):
        """Process the model given the given processor.

        Subclasses of this base class can implement all kind of logic to divide a large dataset in smaller chunks
        (for example slice by slice) and run the processing on each slice separately and join the results afterwards.

        Args:
            processor (ModelProcessor): the processor that defines what to do

        Returns:
            dict: the results as a dictionary of roi lists
        """
        raise NotImplementedError()


class ModelProcessor:

    def process(self, roi_indices, next_indices=None):
        """Get the worker specific for the given voxel indices.

        By adding an additional layer of indirection it is possible for the processing strategy to fine-tune the
        processing of each batch or ROI indices.

        Args:
            roi_indices (ndarray): the list of ROI indices we will use for the current batch
            next_indices (ndarray): the list of ROI indices we will use for the batch after this one. May be None
                if there is no next batch.
        """
        raise NotImplementedError()

    def get_voxels_to_compute(self):
        """Get the ROI indices of the voxels we need to compute.

        This should either return an entire list with all the ROI indices for the given brain mask, or a list
        with the specific roi indices we want the strategy to compute.

        Returns:
            ndarray: the list of ROI indices (indexing the current mask) with the voxels we want to compute.
        """
        raise NotImplementedError()

    def get_total_nmr_voxels(self):
        """Get the total number of voxels that are available for processing.

        This is used for the logging in the processing strategy.

        Returns:
            int: the total number of voxels available for processing
        """
        raise NotImplementedError()

    def combine(self):
        """Combine all the calculated parts.

        Returns:
            the processing results for as much as this is applicable
        """
        raise NotImplementedError()

    def finalize(self):
        """Finalize the processing, added as a convenience function"""
        raise NotImplementedError()


class ChunksProcessingStrategy(ModelProcessingStrategy):

    def __init__(self, *args, **kwargs):
        """This class is a base class for all model fitting strategies that fit the data in chunks/parts."""
        super().__init__()
        self._logger = logging.getLogger(__name__)

    def process(self, processor):
        """Compute all the slices using the implemented chunks generator"""
        chunks = self._get_chunks(processor.get_voxels_to_compute())
        self._process_chunk(processor, chunks)

        self._logger.info('Computed all voxels, now creating nifti\'s')
        return_data = processor.combine()
        processor.finalize()

        return return_data

    def _get_chunks(self, total_roi_indices):
        """Generate the slices/chunks we will use for the fitting.

        Returns:
            lists of ndarray: lists with the voxels to process per chunk
        """
        raise NotImplementedError()

    def _process_chunk(self, processor, chunks):
        """Create the batches.

        The batches contain information about the voxels to process and some meta information like log messages when
        run.
        """
        voxels_processed = 0

        total_roi_indices = processor.get_voxels_to_compute()
        total_nmr_voxels = processor.get_total_nmr_voxels()

        batches = []
        if len(total_roi_indices):
            start_time = timeit.default_timer()
            start_nmr_processed = (total_nmr_voxels - len(total_roi_indices))

            mot_logging_enabled = True
            for chunk_ind, chunk in enumerate(chunks):
                self._logger.info(self._get_batch_start_message(
                        total_nmr_voxels, chunk, total_roi_indices, voxels_processed, start_time, start_nmr_processed))

                next_chunk = None
                if chunk_ind < len(chunks) - 1:
                    next_chunk = chunks[chunk_ind + 1]

                def process():
                    processor.process(chunk, next_indices=next_chunk)

                if mot_logging_enabled:
                    process()
                    mot_logging_enabled = False
                else:
                    with self._with_logging_to_debug():
                        process()

                gc.collect()

                voxels_processed += len(chunk)

            self._logger.info('Computations are at 100%')

        return batches

    @contextmanager
    def _with_logging_to_debug(self):
        package_handlers = [logging.getLogger(package).handlers for package in ['mdt', 'mot']]
        for handlers in package_handlers:
            for handler in handlers:
                handler.setLevel(logging.WARNING)
        yield
        for handlers in package_handlers:
            for handler in handlers:
                handler.setLevel(logging.INFO)

    def _get_batch_start_message(self, total_nmr_voxels, voxel_indices, voxels_to_process, voxels_processed, start_time,
                                 start_nmr_processed):
        total_processed = (total_nmr_voxels - len(voxels_to_process)) + voxels_processed

        def calculate_run_days(runtime):
            if runtime > 24 * 60 * 60:
                return int(runtime // (24. * 60 * 60))
            return 0

        run_time = timeit.default_timer() - start_time
        current_percentage = voxels_processed / (total_nmr_voxels - start_nmr_processed)
        if current_percentage > 0:
            remaining_time = (run_time / current_percentage) - run_time
        else:
            remaining_time = None

        run_time_str = str(calculate_run_days(run_time)) + ':' + time.strftime('%H:%M:%S', time.gmtime(run_time))
        remaining_time_str = (str(calculate_run_days(remaining_time)) + ':' +
                              time.strftime('%H:%M:%S', time.gmtime(remaining_time))) if remaining_time else '?'

        return ('Computations are at {0:.2%}, processing next {1} voxels ('
                '{2} voxels in total, {3} processed). Time spent: {4}, time left: {5} (d:h:m:s).'.
                format(total_processed / total_nmr_voxels,
                       len(voxel_indices),
                       total_nmr_voxels,
                       total_processed,
                       run_time_str,
                       remaining_time_str))


class VoxelRange(ChunksProcessingStrategy):

    def __init__(self, max_nmr_voxels=10000, **kwargs):
        """Optimize a given dataset in batches of the given number of voxels

        Args:
            max_nmr_voxels (int): the number of voxels per batch

        Attributes:
            max_nmr_voxels (int): the number of voxels per chunk
        """
        super().__init__(**kwargs)
        self.nmr_voxels = max_nmr_voxels

    def _get_chunks(self, total_roi_indices):
        chunks = []
        for ind_start in range(0, len(total_roi_indices), self.nmr_voxels):
            ind_end = min(len(total_roi_indices), ind_start + self.nmr_voxels)
            chunks.append(total_roi_indices[ind_start:ind_end])
        return chunks


class SimpleModelProcessor(ModelProcessor):

    def __init__(self, mask, nifti_header, output_dir, tmp_storage_dir, recalculate):
        """A default implementation of a processing worker.

        While the processing strategies determine how to split the work in batches, the workers
        implement the logic on how to process the model. For example, optimization and sample both require
        different processing, while the batch sizes can be determined by a processing strategy.

        Args:
            mask (ndarray): the mask to use during processing
            nifti_header (nibabel nifti header): the nifti header to use for writing the output nifti files
            output_dir (str): the location for the final output files
            tmp_storage_dir (str): the location for the temporary output files
            recalculate (boolean): if we want to recalculate existing results if present
        """
        super().__init__()
        self._write_volumes_gzipped = True
        self._used_mask_name = 'UsedMask'
        self._mask = mask
        self._nifti_header = nifti_header
        self._output_dir = output_dir
        self._tmp_storage_dir = tmp_storage_dir
        self._prepare_tmp_storage(self._tmp_storage_dir, recalculate)
        self._processing_tmp_dir = os.path.join(self._tmp_storage_dir, 'processing_tmp')
        self._roi_lookup_path = os.path.join(self._processing_tmp_dir, 'roi_voxel_lookup_table.npy')
        self._volume_indices = self._create_roi_to_volume_index_lookup_table()
        self._total_nmr_voxels = np.count_nonzero(self._mask)

    def combine(self):
        pass

    def _process(self, roi_indices, next_indices=None):
        """This is the function the user needs to implement to process the dataset.

        Args:
            roi_indices (ndarray): the list of ROI indices we will use for the current batch
            next_indices (ndarray): the list of ROI indices we will use for the batch after this one. May be None
                if there is no next batch.
        """
        raise NotImplementedError()

    def process(self, roi_indices, next_indices=None):
        """By default this will store some information about already processed voxels.

        This will call the user implementable function :meth:`_process` to do the processing.
        """
        self._process(roi_indices, next_indices=next_indices)
        self._write_volumes({'processed_voxels': np.ones(roi_indices.shape[0], dtype=np.bool)},
                            roi_indices, self._processing_tmp_dir)

    def get_voxels_to_compute(self):
        """By default this will return the indices of all the voxels we have not yet computed.

        In the case that recalculate is set to False and we have some intermediate results lying about, this
        function will only return the indices of the voxels we have not yet processed.
        """
        roi_list = np.arange(0, self._total_nmr_voxels)
        processed_voxels_path = os.path.join(self._processing_tmp_dir, 'processed_voxels.npy')
        if os.path.exists(processed_voxels_path):
            return roi_list[np.logical_not(np.squeeze(create_roi(np.load(processed_voxels_path, mmap_mode='r'),
                                                                 self._mask)[roi_list]))]
        return roi_list

    def get_total_nmr_voxels(self):
        """Returns the number of nonzero elements in the mask."""
        return self._total_nmr_voxels

    def finalize(self):
        """Cleans the temporary storage directory."""
        del self._volume_indices
        shutil.rmtree(self._tmp_storage_dir)

    def _prepare_tmp_storage(self, tmp_storage_dir, recalculate):
        if recalculate:
            if os.path.exists(tmp_storage_dir):
                shutil.rmtree(tmp_storage_dir)

        if not os.path.exists(tmp_storage_dir):
            os.makedirs(tmp_storage_dir)

    def _write_volumes(self, results, roi_indices, tmp_dir):
        """Write the result arrays to the temporary storage

        Args:
            results (dict): the dictionary with the results to save
            roi_indices (ndarray): the indices of the voxels we computed
            tmp_dir (str): the directory to save the intermediate results to
        """
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        volume_indices = self._volume_indices[roi_indices, :]

        for param_name, result_array in results.items():
            filename = os.path.join(tmp_dir, param_name + '.npy')
            self._write_volume(result_array, volume_indices, filename)

    def _write_volume(self, data, volume_indices, filename):
        """Write the result of one map to the specified file.

        This is meant to save map data to a temporary .npy file.

        Args:
            data (ndarray): the voxel data to store
            volume_indices (ndarray): the volume indices of the computed data points
            filename (str): the file to write the results to. This by default will append to the file if it exists.
        """
        extra_dims = (1,)
        if len(data.shape) == 2:
            extra_dims = (data.shape[1],)
        elif len(data.shape) > 2:
            extra_dims = data.shape[1:]
        else:
            data = np.reshape(data, (-1, 1))

        mode = 'w+'
        if os.path.isfile(filename):
            mode = 'r+'

        tmp_matrix = open_memmap(filename, mode=mode, dtype=data.dtype,
                                 shape=self._mask.shape[0:3] + extra_dims)
        tmp_matrix[volume_indices[:, 0], volume_indices[:, 1], volume_indices[:, 2]] = data

    def _combine_volumes(self, output_dir, tmp_storage_dir, nifti_header, maps_subdir=''):
        """Combine volumes found in subdirectories to a final volume.

        Args:
            output_dir (str): the location for the output files
            tmp_storage_dir (str): the directory with the temporary results
            maps_subdir (str): the subdirectory for both the output directory as the tmp storage directory.
                If this is set we will load the results from a subdirectory (with this name) from the tmp_storage_dir
                and write the results to a subdirectory (with this name) in the output dir.

        Returns:
            dict: the dictionary with the ROIs for every volume, by parameter name
        """
        full_output_dir = os.path.join(output_dir, maps_subdir)
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        for fname in os.listdir(full_output_dir):
            if fname.endswith('.nii.gz'):
                os.remove(os.path.join(full_output_dir, fname))

        map_names = list(map(lambda p: os.path.splitext(os.path.basename(p))[0],
                             glob.glob(os.path.join(tmp_storage_dir, maps_subdir, '*.npy'))))

        chunks_dir = os.path.join(tmp_storage_dir, maps_subdir)
        for map_name in map_names:
            data = np.load(os.path.join(chunks_dir, map_name + '.npy'), mmap_mode='r')
            write_all_as_nifti({map_name: data}, full_output_dir, nifti_header=nifti_header,
                               gzip=self._write_volumes_gzipped)

    def _create_roi_to_volume_index_lookup_table(self):
        """Creates and returns a lookup table for roi index -> volume index.

        This will create from the given mask a memory mapped lookup table mapping the ROI indices (single integer)
        to the correct voxel location in 3d. To find a voxel index using the ROI index, just index this lookup
        table using the ROI index as index.

        For example, suppose we have the lookup table:

            0: (0, 0, 0)
            1: (0, 0, 1)
            2: (0, 1, 0)
            ...

        We can get the position of a voxel in the 3d space by indexing this array as: ``lookup_table[roi_index]``
        to get the correct 3d location.

        Returns:
            memmap ndarray: the memory mapped array which
        """
        if not os.path.exists(os.path.dirname(self._roi_lookup_path)):
            os.mkdir(os.path.dirname(self._roi_lookup_path))
        if os.path.isfile(self._roi_lookup_path):
            os.remove(self._roi_lookup_path)
        np.save(self._roi_lookup_path, np.argwhere(self._mask))
        return np.load(self._roi_lookup_path, mmap_mode='r')
