"""Contains the processing strategies (and workers) that define how to process a model.

Globally, this module consists out of two public players, the :class:`ModelProcessingStrategy` and the
:class:`ModelProcessor`. The latter, the model processor contains information on how the model needs to be processed.
For example, optimization and sampling are two different operations and hence require their own processing
implementation. Given that the model processor defines how to process the models, the model processing strategy
encapsulates how to process the processors. For example, a strategy may be to split the model into batches and optimize
those while saving intermediate results. More advanced strategies may employ multi-threading to overlap disk write-out
with optimization.
"""
import glob
import hashlib
import logging
import os
import shutil
import timeit
from contextlib import contextmanager

import numpy as np
import time

from mot.cl_routines.optimizing.base import SimpleOptimizationResult
from mot.model_building.model_builders import ParameterTransformedModel
from mot.utils import results_to_dict
import gc
from numpy.lib.format import open_memmap

from mdt.nifti import write_all_as_nifti, get_all_image_data
from mdt.configuration import gzip_optimization_results, gzip_sampling_results
from mdt.utils import create_roi, load_samples

__author__ = 'Robbert Harms'
__date__ = "2016-07-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


DEFAULT_TMP_RESULTS_SUBDIR_NAME = 'tmp_results'


class ModelProcessingStrategy(object):
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


class ChunksProcessingStrategy(ModelProcessingStrategy):

    def __init__(self, *args, **kwargs):
        """This class is a base class for all model slice fitting strategies that fit the data in chunks/parts."""
        super(ChunksProcessingStrategy, self).__init__()
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

        return batches

    @contextmanager
    def _with_logging_to_debug(self):
        handlers = logging.getLogger('mot').handlers
        for handler in handlers:
            handler.setLevel(logging.WARNING)
        yield
        for handler in handlers:
            handler.setLevel(logging.INFO)

    def _get_batch_start_message(self, total_nmr_voxels, voxel_indices, voxels_to_process, voxels_processed, start_time,
                                 start_nmr_processed):
        total_processed = (total_nmr_voxels - len(voxels_to_process)) + voxels_processed

        def calculate_run_days(runtime):
            if runtime > 24 * 60 * 60:
                return runtime // 24 * 60 * 60
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

    def __init__(self, max_nmr_voxels=40000, **kwargs):
        """Optimize a given dataset in batches of the given number of voxels

        Args:
            max_nmr_voxels (int): the number of voxels per batch

        Attributes:
            max_nmr_voxels (int): the number of voxels per chunk
        """
        super(VoxelRange, self).__init__(**kwargs)
        self.nmr_voxels = max_nmr_voxels

    def _get_chunks(self, total_roi_indices):
        chunks = []
        for ind_start in range(0, len(total_roi_indices), self.nmr_voxels):
            ind_end = min(len(total_roi_indices), ind_start + self.nmr_voxels)
            chunks.append(total_roi_indices[ind_start:ind_end])
        return chunks


class ModelProcessor(object):

    def process(self, roi_indices, next_indices=None): #  todo implement
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


class SimpleModelProcessor(ModelProcessor):

    def __init__(self, model, problem_data, output_dir, tmp_storage_dir, clean_tmp_dir, recalculate):
        """A default implementation of a processing worker.

        While the processing strategies determine how to split the work in batches, the workers
        implement the logic on how to process the model. For example, optimization and sampling both require
        different processing, while the batch sizes can be determined by a processing strategy.

        Args:
            model (:class:`~mdt.models.composite.DMRICompositeModel`): the model we want to process
            problem_data (:class:`~mdt.utils.DMRIProblemData`): The problem data object with which
                the model is initialized before running
            output_dir (str): the location for the final output files
            tmp_storage_dir (str): the location for the temporary output files
            clean_tmp_dir (boolean): if we should clean the tmp dir at the end of processing or we can keep it
                for things like memory mapping.
            recalculate (boolean): if we want to recalculate existing results if present
        """
        super(SimpleModelProcessor, self).__init__()
        self._write_volumes_gzipped = True
        self._used_mask_name = 'UsedMask'
        self._model = model
        self._problem_data = problem_data
        self._mask_shape = self._problem_data.mask.shape
        self._output_dir = output_dir
        self._tmp_storage_dir = tmp_storage_dir
        self._prepare_tmp_storage(self._tmp_storage_dir, recalculate)
        self._roi_lookup_path = os.path.join(self._tmp_storage_dir, '_roi_voxel_lookup_table.npy')
        self._clean_tmp_dir = clean_tmp_dir
        self._volume_indices = self._create_roi_to_volume_index_lookup_table()
        self._total_nmr_voxels = np.count_nonzero(self._problem_data.mask)

    def process(self, roi_indices, next_indices=None):
        raise NotImplementedError()

    def get_voxels_to_compute(self):
        roi_list = np.arange(0, self._total_nmr_voxels)
        mask_path = os.path.join(self._tmp_storage_dir, '{}.npy'.format(self._used_mask_name))
        if os.path.exists(mask_path):
            return roi_list[np.logical_not(np.squeeze(create_roi(np.load(mask_path, mmap_mode='r'),
                                                                 self._problem_data.mask)[roi_list]))]
        return roi_list

    def get_total_nmr_voxels(self):
        return self._total_nmr_voxels

    def combine(self):
        del self._volume_indices
        os.remove(self._roi_lookup_path)

    def finalize(self):
        if self._clean_tmp_dir:
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
            storage_path = os.path.join(tmp_dir, param_name + '.npy')

            map_4d_dim_len = 1
            if len(result_array.shape) > 1:
                map_4d_dim_len = result_array.shape[1]
            else:
                result_array = np.reshape(result_array, (-1, 1))

            mode = 'w+'
            if os.path.isfile(storage_path):
                mode = 'r+'
            tmp_matrix = open_memmap(storage_path, mode=mode, dtype=result_array.dtype,
                                     shape=self._mask_shape[0:3] + (map_4d_dim_len,))
            tmp_matrix[volume_indices[:, 0], volume_indices[:, 1], volume_indices[:, 2]] = result_array

        mask_path = os.path.join(tmp_dir, '{}.npy'.format(self._used_mask_name))
        mode = 'w+'
        if os.path.isfile(mask_path):
            mode = 'r+'
        tmp_mask = open_memmap(mask_path, mode=mode, dtype=np.bool, shape=self._mask_shape)
        tmp_mask[volume_indices[:, 0], volume_indices[:, 1], volume_indices[:, 2]] = True

    def _combine_volumes(self, output_dir, tmp_storage_dir, volume_header, maps_subdir=''):
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
        if not os.path.exists(os.path.join(output_dir, maps_subdir)):
            os.makedirs(os.path.join(output_dir, maps_subdir))

        map_names = list(map(lambda p: os.path.splitext(os.path.basename(p))[0],
                             glob.glob(os.path.join(tmp_storage_dir, maps_subdir, '*.npy'))))

        basic_info = (os.path.join(tmp_storage_dir, maps_subdir),
                      os.path.join(output_dir, maps_subdir),
                      volume_header,
                      self._write_volumes_gzipped)
        info_list = [(map_name, basic_info) for map_name in map_names]

        list(map(_combine_volumes_write_out, info_list))

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
        if os.path.isfile(self._roi_lookup_path):
            os.remove(self._roi_lookup_path)
        np.save(self._roi_lookup_path, np.argwhere(self._problem_data.mask))
        return np.load(self._roi_lookup_path, mmap_mode='r')


class FittingProcessor(SimpleModelProcessor):

    def __init__(self, optimizer, model, problem_data, output_dir, tmp_storage_dir, clean_tmp_dir, recalculate):
        """The processing worker for model fitting.

        Use this if you want to use the model processing strategy to do model fitting.

        Args:
            optimizer: the optimization routine to use
        """
        super(FittingProcessor, self).__init__(model, problem_data, output_dir, tmp_storage_dir,
                                               clean_tmp_dir, recalculate)
        self._optimizer = optimizer
        self._write_volumes_gzipped = gzip_optimization_results()

    def process(self, roi_indices, next_indices=None):
        model = self._model.build(roi_indices)
        decorated_model = ParameterTransformedModel(model, self._model.get_parameter_codec())

        optimization_results = self._optimizer.minimize(decorated_model)
        optimization_results = SimpleOptimizationResult(
            model, decorated_model.decode_parameters(optimization_results.get_optimization_result()),
            optimization_results.get_return_codes())

        end_points = optimization_results.get_optimization_result()
        volume_maps = results_to_dict(end_points, model.get_free_param_names())
        volume_maps = model.post_process_optimization_maps(volume_maps, results_array=end_points)
        volume_maps.update({'ReturnCodes': optimization_results.get_return_codes()})
        volume_maps.update(optimization_results.get_error_measures())

        self._write_volumes(volume_maps, roi_indices, self._tmp_storage_dir)

    def combine(self):
        super(FittingProcessor, self).combine()
        self._combine_volumes(self._output_dir, self._tmp_storage_dir, self._problem_data.volume_header)
        return create_roi(get_all_image_data(self._output_dir), self._problem_data.mask)


class SamplingProcessor(SimpleModelProcessor):

    class SampleChainNotStored(object):
        pass

    def __init__(self, sampler, model, problem_data, output_dir, tmp_storage_dir, clean_tmp_dir, recalculate,
                 samples_to_save_method=None, store_volume_maps=True):
        """The processing worker for model sampling.

        Args:
            sampler (AbstractSampler): the optimization sampler to use
            samples_to_save_method (SamplesToSaveMethod): indicates which samples to store
            store_volume_maps (boolean): if we want to store the elements in the 'volume_maps' directory.
                This stores the mean and std maps and some other maps based on the samples.
        """
        super(SamplingProcessor, self).__init__(model, problem_data, output_dir, tmp_storage_dir,
                                                clean_tmp_dir, recalculate)
        self._sampler = sampler
        self._write_volumes_gzipped = gzip_sampling_results()
        self._samples_to_save_method = samples_to_save_method or SaveAllSamples()
        self._store_volume_maps = store_volume_maps

    def get_voxels_to_compute(self):
        """Get the ROI indices of the voxels we need to compute.

        This should either return an entire list with all the ROI indices for the given brain mask, or a list
        with the specific roi indices we want the strategy to compute.

        Returns:
            ndarray: the list of ROI indices (indexing the current mask) with the voxels we want to compute.
        """
        roi_list = np.arange(0, self._total_nmr_voxels)
        samples_paths = glob.glob(os.path.join(self._output_dir, '*.samples.npy'))
        if samples_paths:
            samples_file = samples_paths[0]
            current_results = open_memmap(samples_file, mode='r')
            if current_results.shape[0] == self._total_nmr_voxels:
                mask_path = os.path.join(self._tmp_storage_dir, 'volume_maps', '{}.npy'.format(self._used_mask_name))
                if os.path.exists(mask_path):
                    return roi_list[np.logical_not(np.squeeze(create_roi(np.load(mask_path, mmap_mode='r'),
                                                                         self._problem_data.mask)[roi_list]))]
            del current_results  # force closing memmap
        return roi_list

    def process(self, roi_indices, next_indices=None):
        model = self._model.build(roi_indices)
        sampling_output = self._sampler.sample(model)

        if self._store_volume_maps:
            samples = sampling_output.get_samples()
            volume_maps = model.get_post_sampling_maps(samples)
            self._write_volumes(volume_maps, roi_indices,
                                os.path.join(self._tmp_storage_dir, 'volume_maps'))

            multivariate_statistic = model.get_multivariate_sampling_statistic(samples)
            self._write_volumes(multivariate_statistic, roi_indices,
                                os.path.join(self._tmp_storage_dir, 'multivariate_statistic'))

        self._write_volumes(results_to_dict(sampling_output.get_proposal_state(),
                                            model.get_proposal_state_names()),
                            roi_indices,
                            os.path.join(self._tmp_storage_dir, 'proposal_state'))

        self._tmp_store_mh_state(roi_indices, sampling_output.get_mh_state())

        self._write_volumes(results_to_dict(sampling_output.get_current_chain_position(),
                                            model.get_free_param_names()),
                            roi_indices,
                            os.path.join(self._tmp_storage_dir, 'chain_end_point'))

        if self._samples_to_save_method.store_samples():
            samples = sampling_output.get_samples()
            samples_dict = results_to_dict(samples, model.get_free_param_names())
            self._write_sample_results(samples_dict, roi_indices)

    def combine(self):
        super(SamplingProcessor, self).combine()

        if self._store_volume_maps:
            self._combine_volumes(self._output_dir, self._tmp_storage_dir,
                                  self._problem_data.volume_header, maps_subdir='volume_maps')

        for subdir in ['proposal_state', 'chain_end_point', 'mh_state', 'multivariate_statistic']:
            self._combine_volumes(self._output_dir, self._tmp_storage_dir,
                                  self._problem_data.volume_header, maps_subdir=subdir)

        if self._samples_to_save_method.store_samples():
            return load_samples(self._output_dir)

        return SamplingProcessor.SampleChainNotStored()

    def _write_sample_results(self, results, roi_indices):
        """Write the sample results to a .npy file.

        If the given sample files do not exists or if the existing file is not large enough it will create one
        with enough storage to hold all the samples for the given total_nmr_voxels.
        On storing it should also be given a list of voxel indices with the indices of the voxels that are being stored.

        Args:
            results (dict): the samples to write
            roi_indices (ndarray): the roi indices of the voxels we computed
        """
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        for fname in os.listdir(self._output_dir):
            if fname.endswith('.samples.npy'):
                chain_name = fname[0:-len('.samples.npy')]
                if chain_name not in results:
                    os.remove(os.path.join(self._output_dir, fname))

        save_indices = self._samples_to_save_method.indices_to_store(results[list(results.keys())[0]].shape[1])

        for map_name, samples in results.items():
            samples_path = os.path.join(self._output_dir, map_name + '.samples.npy')
            mode = 'w+'

            if os.path.isfile(samples_path):
                mode = 'r+'
                current_results = open_memmap(samples_path, mode='r')
                if current_results.shape[1] != len(save_indices):
                    mode = 'w+'
                del current_results  # closes the memmap

            saved = open_memmap(samples_path, mode=mode, dtype=samples.dtype,
                                shape=(self._total_nmr_voxels, len(save_indices)))
            saved[roi_indices, :] = samples[:, save_indices]
            del saved

    def _tmp_store_mh_state(self, roi_indices, mh_state):
        items = {'proposal_state_sampling_counter': mh_state.get_proposal_state_sampling_counter(),
                 'proposal_state_acceptance_counter': mh_state.get_proposal_state_acceptance_counter(),
                 'online_parameter_variance': mh_state.get_online_parameter_variance(),
                 'online_parameter_variance_update_m2': mh_state.get_online_parameter_variance_update_m2(),
                 'online_parameter_mean': mh_state.get_online_parameter_mean(),
                 'rng_state': mh_state.get_rng_state()}

        self._write_volumes(items, roi_indices, os.path.join(self._tmp_storage_dir, 'mh_state'))

        if not os.path.isdir(os.path.join(self._output_dir, 'mh_state')):
            os.makedirs(os.path.join(self._output_dir, 'mh_state'))
        with open(os.path.join(self._output_dir, 'mh_state', 'nmr_samples_drawn.txt'), 'w') as f:
            f.write(str(mh_state.nmr_samples_drawn))


class SamplesToSaveMethod(object):
    """Defines if and how many samples are being stored.

    This can be used to only save a subset of the calculated samples, while still using the entire chain for
    the point estimates. This is the ideal combination of high estimate accuracy and storage.
    """

    def store_samples(self):
        """If we should store the samples at all

        Returns:
            boolean: if we should store the samples
        """
        raise NotImplementedError()

    def indices_to_store(self, nmr_samples):
        """Return indices that indicate how many and which samples to store.

        The return indices should be between 0 and nmr_samples

        Args:
            nmr_samples (int): the maximum number of samples

        Returns:
            ndarray: indices of the samples to store
        """
        raise NotImplementedError()


class SaveAllSamples(SamplesToSaveMethod):
    """Indicates that all the samples should be saved."""

    def store_samples(self):
        return True

    def indices_to_store(self, nmr_samples):
        return np.arange(nmr_samples)


class SaveThinnedSamples(SamplesToSaveMethod):

    def __init__(self, thinning):
        """Indicates that only every n sample should be saved.

        For example, if thinning = 1 we save every sample. If thinning = 2 we save every other sample, etc.

        Args:
            thinning (int): the thinning factor to apply
        """
        self._thinning = thinning

    def store_samples(self):
        return True

    def indices_to_store(self, nmr_samples):
        return np.arange(0, nmr_samples, self._thinning)


class SaveNoSamples(SamplesToSaveMethod):
    """Indicates that no samples should be saved."""

    def store_samples(self):
        return False

    def indices_to_store(self, nmr_samples):
        return np.array([])


def get_full_tmp_results_path(output_dir, tmp_dir):
    """Get a temporary results path for processing.

    Args:
        output_dir (str): the output directory of the results
        tmp_dir (str): a preferred tmp dir. If not given we create a temporary directory in the output_dir.

    Returns:
        str: a temporary results path for saving computation results
    """
    if tmp_dir is None:
        return os.path.join(output_dir, DEFAULT_TMP_RESULTS_SUBDIR_NAME)

    if not output_dir.endswith('/'):
        output_dir += '/'

    return os.path.join(tmp_dir, hashlib.md5(output_dir.encode('utf-8')).hexdigest())


def _combine_volumes_write_out(info_pair):
    """Write out the given information to a nifti volume.

    Needs to be used by ModelProcessor._combine_volumes
    """
    map_name, info_list = info_pair
    chunks_dir, output_dir, volume_header, write_gzipped = info_list

    data = np.load(os.path.join(chunks_dir, map_name + '.npy'), mmap_mode='r')
    write_all_as_nifti({map_name: data}, output_dir, volume_header, gzip=write_gzipped)
