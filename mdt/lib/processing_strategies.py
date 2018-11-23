"""Contains the processing strategies (and workers) that define how to process a model (fitting and sample).

Globally, this module consists out of two public players, the :class:`ModelProcessingStrategy` and the
:class:`ModelProcessor`. The latter, the model processor contains information on how the model needs to be processed.
For example, optimization and sample are two different operations and hence require their own processing
implementation. Given that the model processor defines how to process the models, the model processing strategy
encapsulates how to process the processors. For example, a strategy may be to split all voxels into batches and optimize
those while saving intermediate results.
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

import gc
from numpy.lib.format import open_memmap

import mot
from mdt.lib.fsl_sampling_routine import FSLSamplingRoutine
from mdt.lib.nifti import write_all_as_nifti, get_all_nifti_data
from mdt.configuration import gzip_optimization_results, gzip_sampling_results
from mdt.utils import create_roi, load_samples
import collections

from mot.sample import AdaptiveMetropolisWithinGibbs, SingleComponentAdaptiveMetropolis
from mdt.model_building.utils import ObjectiveFunctionWrapper
from mot.configuration import CLRuntimeInfo
from mot.optimize import minimize
from mot.sample.mwg import MetropolisWithinGibbs
from mot.sample.t_walk import ThoughtfulWalk

__author__ = 'Robbert Harms'
__date__ = "2016-07-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


DEFAULT_TMP_RESULTS_SUBDIR_NAME = 'tmp_results'


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


class ChunksProcessingStrategy(ModelProcessingStrategy):

    def __init__(self, *args, **kwargs):
        """This class is a base class for all model slice fitting strategies that fit the data in chunks/parts."""
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


class FittingProcessor(SimpleModelProcessor):

    def __init__(self, method, model, mask, nifti_header, output_dir, tmp_storage_dir, recalculate,
                 optimizer_options=None):
        """The processing worker for model fitting.

        Use this if you want to use the model processing strategy to do model fitting.

        Args:
            method: the optimization routine to use
        """
        super().__init__(mask, nifti_header, output_dir, tmp_storage_dir, recalculate)
        self._model = model
        self._method = method
        self._optimizer_options = optimizer_options
        self._write_volumes_gzipped = gzip_optimization_results()
        self._subdirs = set()
        self._logger=logging.getLogger(__name__)

    def _process(self, roi_indices, next_indices=None):
        with self._model.voxels_to_analyze_context(roi_indices):
            codec = self._model.get_parameter_codec()

            cl_runtime_info = CLRuntimeInfo()

            self._logger.info('Starting optimization')
            self._logger.info('Using MOT version {}'.format(mot.__version__))
            self._logger.info('We will use a {} precision float type for the calculations.'.format(
                'double' if cl_runtime_info.double_precision else 'single'))
            for env in cl_runtime_info.cl_environments:
                self._logger.info('Using device \'{}\'.'.format(str(env)))
            self._logger.info('Using compile flags: {}'.format(cl_runtime_info.compile_flags))

            if self._optimizer_options:
                self._logger.info('We will use the optimizer {} '
                                  'with optimizer settings {}'.format(self._method, self._optimizer_options))
            else:
                self._logger.info('We will use the optimizer {} with default settings.'.format(self._method))

            x0 = codec.encode(self._model.get_initial_parameters(), self._model.get_kernel_data())
            lower_bounds, upper_bounds = codec.encode_bounds(self._model.get_lower_bounds(),
                                                             self._model.get_upper_bounds())

            wrapper = ObjectiveFunctionWrapper(x0.shape[1])
            objective_func = wrapper.wrap_objective_function(self._model.get_objective_function(),
                                                             codec.get_decode_function())
            input_data = wrapper.wrap_input_data(self._model.get_kernel_data())

            results = minimize(objective_func, x0, method=self._method,
                               nmr_observations=self._model.get_nmr_observations(),
                               cl_runtime_info=cl_runtime_info,
                               data=input_data,
                               lower_bounds=lower_bounds,
                               upper_bounds=upper_bounds,
                               options=self._optimizer_options)

            self._logger.info('Finished optimization')
            self._logger.info('Starting post-processing')

            x_final = codec.decode(results['x'], self._model.get_kernel_data())

            results = self._model.get_post_optimization_output(x_final, results['status'])
            results.update({self._used_mask_name: np.ones(roi_indices.shape[0], dtype=np.bool)})

            self._logger.info('Finished post-processing')

            self._write_output_recursive(results, roi_indices)

    def _write_output_recursive(self, results, roi_indices, sub_dir=''):
        current_output = {}
        sub_dir = sub_dir

        for key, value in results.items():
            if isinstance(value, collections.Mapping):
                self._write_output_recursive(value, roi_indices, os.path.join(sub_dir, key))
            else:
                current_output[key] = value

        self._write_volumes(current_output, roi_indices, os.path.join(self._tmp_storage_dir, sub_dir))
        self._subdirs.add(sub_dir)

    def combine(self):
        super().combine()
        for subdir in self._subdirs:
            self._combine_volumes(self._output_dir, self._tmp_storage_dir,
                                  self._nifti_header, maps_subdir=subdir)
        return create_roi(get_all_nifti_data(self._output_dir), self._mask)


class SamplingProcessor(SimpleModelProcessor):

    class SampleChainNotStored:
        pass

    def __init__(self, nmr_samples, thinning, burnin, method, model, mask, nifti_header, output_dir, tmp_storage_dir,
                 recalculate, samples_storage_strategy=None, post_sampling_cb=None, sampler_options=None):
        """The processing worker for model sample.

        Args:
            nmr_samples (int): the number of samples we would like to return.
            burnin (int): the number of samples to burn-in, that is, to discard before returning the desired
                number of samples
            thinning (int): how many sample we wait before storing a new one. This will draw extra samples such that
                    the total number of samples generated is ``nmr_samples * (thinning)`` and the number of samples
                    stored is ``nmr_samples``. If set to one or lower we store every sample after the burn in.
            method (str): The sampling method to use, one of:
                - 'AMWG', for the Adaptive Metropolis-Within-Gibbs method
                - 'SCAM', for the Single Component Adaptive Metropolis
                - 'FSL', for the sampling method used in the FSL toolbox
            samples_storage_strategy (SamplesStorageStrategy): indicates which samples to store
            post_sampling_cb (Callable[
                [mot.sample.base.SamplingOutput, mdt.models.composite.DMRICompositeModel], Optional[Dict]]):
                    additional post-processing called after sampling. This function can optionally return a (nested)
                    dictionary with as keys dir-/file-names and as values maps to be stored in the results directory.
            sampler_options (dict): specific options for the MCMC routine. These will be provided to the sampling routine
                as additional keyword arguments to the constructor.
        """
        super().__init__(mask, nifti_header, output_dir, tmp_storage_dir, recalculate)
        self._nmr_samples = nmr_samples
        self._thinning = thinning
        self._burnin = burnin
        self._method = method
        self._model = model
        self._write_volumes_gzipped = gzip_sampling_results()
        self._samples_to_save_method = samples_storage_strategy or SaveAllSamples()
        self._subdirs = set()
        self._logger = logging.getLogger(__name__)
        self._samples_output_stored = []
        self._post_sampling_cb = post_sampling_cb
        self._sampler_options = sampler_options or {}

    def _process(self, roi_indices, next_indices=None):
        with self._model.voxels_to_analyze_context(roi_indices):

            method = None
            method_args = [self._model.get_log_likelihood_function(),
                           self._model.get_log_prior_function(),
                           self._model.get_initial_parameters()]
            method_kwargs = {'data': self._model.get_kernel_data()}

            if self._method in ['AMWG', 'SCAM', 'MWG', 'FSL']:
                method_args.append(self._model.get_rwm_proposal_stds())
                method_kwargs.update(finalize_proposal_func=self._model.get_finalize_proposal_function())

            if self._method == 'AMWG':
                method = AdaptiveMetropolisWithinGibbs
            elif self._method == 'SCAM':
                method = SingleComponentAdaptiveMetropolis
                method_kwargs['epsilon'] = self._model.get_rwm_epsilons()
            elif self._method == 'MWG':
                method = MetropolisWithinGibbs
            elif self._method == 'FSL':
                method = FSLSamplingRoutine
            elif self._method == 't-walk':
                method = ThoughtfulWalk
                method_args.append(self._model.get_random_parameter_positions()[..., 0])
                method_kwargs.update(finalize_proposal_func=self._model.get_finalize_proposal_function())

            method_kwargs.update(self._sampler_options)

            if method is None:
                raise ValueError('Could not find the sampler with name {}.'.format(self._method))

            sampler = method(*method_args, **method_kwargs)
            sampling_output = sampler.sample(self._nmr_samples, burnin=self._burnin, thinning=self._thinning)
            samples = sampling_output.get_samples()

            self._logger.info('Starting post-processing')
            maps_to_save = self._model.get_post_sampling_maps(sampling_output)
            maps_to_save.update({self._used_mask_name: np.ones(samples.shape[0], dtype=np.bool)})

            if self._post_sampling_cb:
                out = self._post_sampling_cb(sampling_output, self._model)
                if out:
                    maps_to_save.update(out)

            self._write_output_recursive(maps_to_save, roi_indices)

            def get_output(output_name):
                if output_name in self._model.get_free_param_names():
                    return samples[:, ind, ...]
                elif output_name == 'LogLikelihood':
                    return sampling_output.get_log_likelihoods()
                elif output_name == 'LogPrior':
                    return sampling_output.get_log_priors()

            items_to_save = {}
            for ind, name in enumerate(list(self._model.get_free_param_names()) + ['LogLikelihood', 'LogPrior']):
                if self._samples_to_save_method.store_samples(name):
                    self._samples_output_stored.append(name)
                    items_to_save.update({name: get_output(name)})
            self._write_sample_results(items_to_save, roi_indices)

            self._logger.info('Finished post-processing')

    def combine(self):
        super().combine()

        for subdir in self._subdirs:
            self._combine_volumes(self._output_dir, self._tmp_storage_dir,
                                  self._nifti_header, maps_subdir=subdir)

        if self._samples_output_stored:
            return load_samples(self._output_dir)

        return SamplingProcessor.SampleChainNotStored()

    def _write_output_recursive(self, results, roi_indices, sub_dir=''):
        current_output = {}
        sub_dir = sub_dir

        for key, value in results.items():
            if isinstance(value, collections.Mapping):
                self._write_output_recursive(value, roi_indices, os.path.join(sub_dir, key))
            else:
                current_output[key] = value

        self._write_volumes(current_output, roi_indices, os.path.join(self._tmp_storage_dir, sub_dir))
        self._subdirs.add(sub_dir)

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

        for output_name, samples in results.items():
            save_indices = self._samples_to_save_method.indices_to_store(output_name, samples.shape[1])
            samples_path = os.path.join(self._output_dir, output_name + '.samples.npy')
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


class SamplesStorageStrategy:
    """Defines if and how many samples are being stored, per output item.

    This can be used to only save a subset of the calculated samples, while still using the entire chain for
    the point estimates. This is the ideal combination of high accuracy estimation and storage.

    Additionally, this can be used to store only specific output items instead of all.
    """

    def store_samples(self, output_name):
        """If we should store the samples of this output.

        The outputs are model parameters and additional maps like log-likelihoods.

        Args:
            output_name (str): the name of the output item we want to store the samples of

        Returns:
            boolean: if we should store the samples of this output element.
        """
        raise NotImplementedError()

    def indices_to_store(self, output_name, nmr_samples):
        """Return the indices of the samples to store for this map.

        The return indices should be between 0 and nmr_samples

        Args:
            output_name (str): the name of the output item we want to store the samples of
            nmr_samples (int): the maximum number of samples

        Returns:
            ndarray: indices of the samples to store
        """
        raise NotImplementedError()


class SaveSpecificMaps(SamplesStorageStrategy):

    def __init__(self, included=None, excluded=None):
        """A saving strategy that will only store the output items specified.

        If a list of included items is given, we will only store the items that are included in that list. If a list
        of excluded items are given we will store every item that is not in that list. These two options are mutually
        exclusive, i.e. use only one of the two.

        Args:
            included (list): store only the items in this list
            excluded (list): store all items not in this list
        """
        self._included = included
        self._excluded = excluded

        if self._included and self._excluded:
            raise ValueError('Can not specify both inclusion and exclusion items.')

    def store_samples(self, output_name):
        if self._included:
            return output_name in self._included
        if self._excluded:
            return output_name not in self._excluded

    def indices_to_store(self, output_name, nmr_samples):
        return np.arange(nmr_samples)


class SaveAllSamples(SamplesStorageStrategy):
    """Indicates that all the samples should be saved."""

    def store_samples(self, output_name):
        return True

    def indices_to_store(self, output_name, nmr_samples):
        return np.arange(nmr_samples)


class SaveThinnedSamples(SamplesStorageStrategy):

    def __init__(self, thinning):
        """Indicates that only every n sample should be saved.

        For example, if thinning = 1 we save every sample. If thinning = 2 we save every other sample, etc.

        Args:
            thinning (int): the thinning factor to apply
        """
        self._thinning = thinning

    def store_samples(self, output_name):
        return True

    def indices_to_store(self, output_name, nmr_samples):
        return np.arange(0, nmr_samples, self._thinning)


class SaveNoSamples(SamplesStorageStrategy):
    """Indicates that no samples should be saved."""

    def store_samples(self, output_name):
        return False

    def indices_to_store(self, output_name, nmr_samples):
        return np.array([])


class SaveSpecificSamples(SamplesStorageStrategy):

    def __init__(self, sample_indices):
        """Save all the samples at the specified indices.

        Args:
            sample_indices (sequence): the list of indices we want to save.
        """
        self._sample_indices = list(sample_indices)

    def store_samples(self, output_name):
        return len(self._sample_indices)

    def indices_to_store(self, output_name, nmr_samples):
        return self._sample_indices


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
    chunks_dir, output_dir, nifti_header, write_gzipped = info_list

    data = np.load(os.path.join(chunks_dir, map_name + '.npy'), mmap_mode='r')
    write_all_as_nifti({map_name: data}, output_dir, nifti_header=nifti_header, gzip=write_gzipped)
