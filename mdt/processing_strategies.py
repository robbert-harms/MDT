import glob
import hashlib
import logging
import os
import shutil
import timeit
from contextlib import contextmanager

import numpy as np
import time
from numpy.lib.format import open_memmap

from mdt.nifti import write_all_as_nifti, get_all_image_data
from mdt.configuration import gzip_optimization_results, gzip_sampling_results
from mdt.utils import create_roi, load_samples

__author__ = 'Robbert Harms'
__date__ = "2016-07-29"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelProcessingStrategy(object):

    def __init__(self, tmp_dir=None):
        """Model processing strategies define in how many parts a composite model is processed.

        This uses the :attr:`~mot.model_building.model_builders.OptimizeModelBuilder.problems_to_analyze` attribute
        of the MOT model builder (:class:`mot.model_building.model_builders.OptimizeModelBuilder`) to select the
        voxels to process. That attribute arranges that only a selection of the problems are analyzed
        instead of all of them. We set the range, optimize, collect the data and then repeat until all the voxels
        are optimized.

        Args:
            tmp_dir (str): The temporary dir for the calculations. If set to None we write the temporary results in the
                results folder of each subject. Else, if set set to a specific path we will store the temporary results
                in a subfolder in the given folder (the subfolder will be a hash of the original folder).
        """
        self._logger = logging.getLogger(__name__)
        self._tmp_dir = tmp_dir

    def set_tmp_dir(self, tmp_dir):
        """Set the temporary directory for the calculations. This overwrites the current value.

        Args:
            tmp_dir (str): The temporary dir for the calculations. If set to None we write the temporary results in the
                results folder of each subject. Else, if set to a specific path we will store the temporary results
                in a subfolder in the given folder (the subfolder will be a hash of the original folder).
        """
        self._tmp_dir = tmp_dir
        return self

    def run(self, model, problem_data, output_path, recalculate, worker_generator):
        """Process the given dataset using the logistics the subclass.

        Subclasses of this base class can implement all kind of logic to divide a large dataset in smaller chunks
        (for example slice by slice) and run the processing on each slice separately and join the results afterwards.

        Args:
             model (:class:`~mdt.models.composite.DMRICompositeModel`): the model we want to process
             problem_data (:class:`~mdt.utils.DMRIProblemData`): The problem data object with which
                the model is initialized before running
             output_path (string): The full path to the folder where to place the output
             recalculate (boolean): If we want to recalculate the results if they are already present.
             worker_generator (ModelProcessingWorkerCreator): the generator for creating the worker we will use

        Returns:
            dict: the results as a dictionary of roi lists
        """


class SimpleProcessingStrategy(ModelProcessingStrategy):

    def __init__(self, tmp_dir=None, honor_voxels_to_analyze=True):
        """This class is a base class for all model slice fitting strategies that fit the data in chunks/parts.

        Args:
            honor_voxels_to_analyze (bool): if set to True, we use the model's voxels_to_analyze setting if set
                instead of fitting all voxels in the mask
        """
        super(SimpleProcessingStrategy, self).__init__(tmp_dir=tmp_dir)
        self._honor_voxels_to_analyze = honor_voxels_to_analyze

    @contextmanager
    def _tmp_storage_dir(self, model_output_path, recalculate):
        """Creates a temporary storage dir for the calculations. Removes the dir after calculations.

        Use this manager as a context for running the calculations.

        Args:
            model_output_path (str): the output path of the final model results. We use this to create the tmp_dir.
             recalculate (boolean): if true and the data exists, we throw everything away to start over.
        """
        tmp_storage_dir = self._get_tmp_results_dir(model_output_path)
        self._prepare_tmp_storage_dir(tmp_storage_dir, recalculate)
        yield tmp_storage_dir
        shutil.rmtree(tmp_storage_dir)

    def _get_tmp_results_dir(self, model_output_path):
        """Get the temporary results dir we need to use for processing.

        If self._tmp_dir is set to a non null value we will use a subdirectory in self._tmp_dir.
        Else, if self._tmp_dir is null, we will use a subdir of the model_output_path.

        Args:
            model_output_path (str): the output path of the final model results.

        Returns:
            str: the path to store the temporary results in
        """
        if self._tmp_dir is None:
            return os.path.join(model_output_path, 'tmp_results')

        self._logger.info('Using user defined path for saving the temporary results: {}.'.format(self._tmp_dir))
        return os.path.join(self._tmp_dir, hashlib.md5(model_output_path.encode('utf-8')).hexdigest())

    @staticmethod
    def _prepare_tmp_storage_dir(tmp_storage_dir, recalculate):
        """Prepare the directory for the temporary storage.

        If recalculate is set to True we will remove the storage dir if it exists. Else if False, we will create the
        dir if it does not exist.

        Args:
            tmp_storage_dir (str): the full path to the chunks directory.
            recalculate (boolean): if true and the data exists, we throw everything away to start over.
        """
        if recalculate:
            if os.path.exists(tmp_storage_dir):
                shutil.rmtree(tmp_storage_dir)

        if not os.path.exists(tmp_storage_dir):
            os.makedirs(tmp_storage_dir)

    @contextmanager
    def _selected_indices(self, model, chunk_indices):
        """Create a context in which problems_to_analyze attribute of the models is set to the selected indices.

        Args:
            model: the model to which to set the problems_to_analyze
            chunk_indices (ndarray): the list of voxel indices we want to use for processing
        """
        old_setting = model.problems_to_analyze
        model.problems_to_analyze = chunk_indices
        yield
        model.problems_to_analyze = old_setting


class ChunksProcessingStrategy(SimpleProcessingStrategy):

    def run(self, model, problem_data, output_path, recalculate, worker_generator):
        """Compute all the slices using the implemented chunks generator"""
        with self._tmp_storage_dir(output_path, recalculate) as tmp_storage_dir:
            voxels_processed = 0

            worker = worker_generator.create_worker(model, problem_data, output_path,
                                                    tmp_storage_dir, self._honor_voxels_to_analyze)

            total_roi_indices = worker.get_voxels_to_compute()
            if len(total_roi_indices):

                start_time = timeit.default_timer()
                start_nmr_processed = (np.count_nonzero(problem_data.mask) - len(total_roi_indices))

                for chunk_indices in self._chunks_generator(model, problem_data, output_path, worker,
                                                            total_roi_indices):
                    with self._selected_indices(model, chunk_indices):
                        self._run_on_chunk(problem_data, worker, chunk_indices, total_roi_indices,
                                           voxels_processed, start_time, start_nmr_processed)

                    voxels_processed += len(chunk_indices)

            self._logger.info('Computed all voxels, now creating nifti\'s')
            return_data = worker.combine()

        return return_data

    def _chunks_generator(self, model, problem_data, output_path, worker, total_roi_indices):
        """Generate the slices/chunks we will use for the fitting.

        Yields:
            ndarray: the roi indices per chunk we want to process
        """
        raise NotImplementedError

    def _run_on_chunk(self, problem_data, worker, voxel_indices, voxels_to_process, voxels_processed, start_time,
                      start_nmr_processed):
        """Run the worker on the given chunk."""
        total_nmr_voxels = np.count_nonzero(problem_data.mask)
        total_processed = (total_nmr_voxels - len(voxels_to_process)) + voxels_processed

        run_time = timeit.default_timer() - start_time
        current_percentage = voxels_processed / (total_nmr_voxels - start_nmr_processed)
        if current_percentage > 0:
            remaining_time = (run_time / current_percentage) - run_time
        else:
            remaining_time = None

        self._logger.info('Computations are at {0:.2%}, processing next {1} voxels ('
                          '{2} voxels in total, {3} processed). Time spent: {4}, time left: {5} (h:m:s).'.
                          format(total_processed / total_nmr_voxels, len(voxel_indices),
                                 total_nmr_voxels,
                                 total_processed,
                                 time.strftime('%H:%M:%S', time.gmtime(run_time)),
                                 time.strftime('%H:%M:%S', time.gmtime(remaining_time)) if remaining_time else '?'))

        worker.process(voxel_indices)


class ModelProcessingWorkerCreator(object):

    def create_worker(self, model, problem_data, output_dir, tmp_storage_dir, honor_voxels_to_analyze):
        """Create and return the worker that the processing strategy will use.

        Args:
            model (:class:`~mdt.models.composite.DMRICompositeModel`): the model we want to process
            problem_data (:class:`~mdt.utils.DMRIProblemData`): The problem data object with which
                the model is initialized before running
            output_dir (str): the location for the final output files
            tmp_storage_dir (str): the location for the temporary output files
            honor_voxels_to_analyze (boolean): if we should honor the voxels_to_analyze list in the model if applicable.

        Returns:
            ModelProcessingWorker: the worker the processing strategy will use.
        """


class SimpleModelProcessingWorkerGenerator(ModelProcessingWorkerCreator):

    def __init__(self, callback_function):
        """Create a generator that will instantiate the worker using the given callback function.

        Args:
            callback_function: the callback function we will call when create_worker is called.
        """
        self._callback_function = callback_function

    def create_worker(self, model, problem_data, output_dir, tmp_storage_dir, honor_voxels_to_analyze):
        return self._callback_function(model, problem_data, output_dir, tmp_storage_dir, honor_voxels_to_analyze)


class ModelProcessingWorker(object):

    def __init__(self, model, problem_data, output_dir, tmp_storage_dir, honor_voxels_to_analyze):
        self._write_volumes_gzipped = True
        self._used_mask_name = 'UsedMask'
        self._model = model
        self._problem_data = problem_data
        self._output_dir = output_dir
        self._tmp_storage_dir = tmp_storage_dir
        self._honor_voxels_to_analyze = honor_voxels_to_analyze
        self._roi_lookup_path = os.path.join(self._tmp_storage_dir, '_roi_voxel_lookup_table.npy')
        self._volume_indices = self._create_roi_to_volume_index_lookup_table()

    def process(self, roi_indices):
        """Process the indicated voxels in the way prescribed by this worker.

        Since the processing strategy can use all voxels to do the analysis in one go, this function
        should return all the output it can, i.e. the same kind of output as from the function :meth:`combine`.

        Args:
            roi_indices (ndarray): The list of roi indices we want to compute

        Returns:
            the results for this single processing step
        """

    def get_voxels_to_compute(self):
        """Get the ROI indices of the voxels we need to compute.

        This should either return an entire list with all the ROI indices for the given brain mask, or a list
        with the specific roi indices we want the strategy to compute.

        Returns:
            ndarray: the list of ROI indices (indexing the current mask) with the voxels we want to compute.
        """
        if self._honor_voxels_to_analyze and self._model.problems_to_analyze:
            roi_list = self._model.problems_to_analyze
        else:
            roi_list = np.arange(0, np.count_nonzero(self._problem_data.mask))

        mask_path = os.path.join(self._tmp_storage_dir, '{}.npy'.format(self._used_mask_name))
        if os.path.exists(mask_path):
            return roi_list[np.logical_not(np.squeeze(create_roi(np.load(mask_path, mmap_mode='r'),
                                                                 self._problem_data.mask)[roi_list]))]
        return roi_list

    def combine(self):
        """Combine all the calculated parts.

        Returns:
            the processing results for as much as this is applicable
        """
        del self._volume_indices
        os.remove(self._roi_lookup_path)

    def _write_volumes(self, roi_indices, results, tmp_dir):
        """Write the result arrays to the temporary storage

        Args:
            roi_indices (ndarray): the indices of the voxels we computed
            results (dict): the dictionary with the results to save
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
                                     shape=self._problem_data.mask.shape[0:3] + (map_4d_dim_len,))
            tmp_matrix[volume_indices[:, 0], volume_indices[:, 1], volume_indices[:, 2]] = result_array

        mask_path = os.path.join(tmp_dir, '{}.npy'.format(self._used_mask_name))
        mode = 'w+'
        if os.path.isfile(mask_path):
            mode = 'r+'
        tmp_mask = open_memmap(mask_path, mode=mode, dtype=np.bool, shape=self._problem_data.mask.shape)
        tmp_mask[volume_indices[:, 0], volume_indices[:, 1], volume_indices[:, 2]] = True

    def _combine_volumes(self, output_dir, chunks_dir, volume_header, maps_subdir=''):
        """Combine volumes found in subdirectories to a final volume.

        Args:
            output_dir (str): the location for the output files
            chunks_dir (str): the directory in which all the chunks are located
            maps_subdir (str): we may have per chunk a subdirectory in which the maps are located. This
                parameter is for that subdir. Example search: <chunks_dir>/<chunk>/<maps_subdir>/*.nii*

        Returns:
            dict: the dictionary with the ROIs for every volume, by parameter name
        """
        if not os.path.exists(os.path.join(output_dir, maps_subdir)):
            os.makedirs(os.path.join(output_dir, maps_subdir))

        map_names = list(map(lambda p: os.path.splitext(os.path.basename(p))[0],
                             glob.glob(os.path.join(chunks_dir, maps_subdir, '*.npy'))))

        basic_info = (os.path.join(chunks_dir, maps_subdir),
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


def _combine_volumes_write_out(info_pair):
    """Write out the given information to a nifti volume.

    Needs to be used by ModelProcessingWorker._combine_volumes
    """
    map_name, info_list = info_pair
    chunks_dir, output_dir, volume_header, write_gzipped = info_list

    data = np.load(os.path.join(chunks_dir, map_name + '.npy'), mmap_mode='r')
    write_all_as_nifti({map_name: data}, output_dir, volume_header, gzip=write_gzipped)
    del data  # force closing memmap


class FittingProcessingWorker(ModelProcessingWorker):

    def __init__(self, optimizer, *args):
        """The processing worker for model fitting.

        Use this if you want to use the model processing strategy to do model fitting.

        Args:
            optimizer: the optimization routine to use
        """
        super(FittingProcessingWorker, self).__init__(*args)
        self._optimizer = optimizer
        self._write_volumes_gzipped = gzip_optimization_results()

    def process(self, roi_indices):
        results, extra_output = self._optimizer.minimize(self._model, full_output=True)
        results.update(extra_output)

        self._write_volumes(roi_indices, results, self._tmp_storage_dir)
        return results

    def combine(self):
        super(FittingProcessingWorker, self).combine()
        self._combine_volumes(self._output_dir, self._tmp_storage_dir, self._problem_data.volume_header)
        return create_roi(get_all_image_data(self._output_dir), self._problem_data.mask)


class SamplingProcessingWorker(ModelProcessingWorker):

    class SampleChainNotStored(object):
        pass

    def __init__(self, sampler, store_samples=False, store_volume_maps=True, *args):
        """The processing worker for model sampling.

        Use this if you want to use the model processing strategy to do model sampling.

        Args:
            sampler (AbstractSampler): the optimization sampler to use
            store_samples (boolean): if set to False we will store none of the samples. Use this
                if you are only interested in the volume maps and not in the entire sample chain.
                If set to True the process and combine function will no longer return any results.
            store_volume_maps (boolean): if we want to store the elements in the 'volume_maps' directory.
                This stores the mean and std maps and some other maps based on the samples.
        """
        super(SamplingProcessingWorker, self).__init__(*args)
        self._sampler = sampler
        self._write_volumes_gzipped = gzip_sampling_results()
        self._store_samples = store_samples
        self._store_volume_maps = store_volume_maps

    def get_voxels_to_compute(self):
        """Get the ROI indices of the voxels we need to compute.

        This should either return an entire list with all the ROI indices for the given brain mask, or a list
        with the specific roi indices we want the strategy to compute.

        Returns:
            ndarray: the list of ROI indices (indexing the current mask) with the voxels we want to compute.
        """
        if self._honor_voxels_to_analyze and self._model.problems_to_analyze:
            roi_list = self._model.problems_to_analyze
        else:
            roi_list = np.arange(0, np.count_nonzero(self._problem_data.mask))

        samples_paths = glob.glob(os.path.join(self._output_dir, '*.samples.npy'))
        if samples_paths:
            samples_file = samples_paths[0]
            current_results = open_memmap(samples_file, mode='r')
            if current_results.shape[0] == np.count_nonzero(self._problem_data.mask):
                mask_path = os.path.join(self._tmp_storage_dir, 'volume_maps', '{}.npy'.format(self._used_mask_name))
                if os.path.exists(mask_path):
                    return roi_list[np.logical_not(np.squeeze(create_roi(np.load(mask_path, mmap_mode='r'),
                                                                         self._problem_data.mask)[roi_list]))]
            del current_results  # force closing memmap
        return roi_list

    def process(self, roi_indices):
        results, volume_maps, proposal_state = self._sampler.sample(self._model, full_output=True)

        if self._store_volume_maps:
            self._write_volumes(roi_indices, volume_maps, os.path.join(self._tmp_storage_dir, 'volume_maps'))

        self._write_volumes(roi_indices, proposal_state, os.path.join(self._tmp_storage_dir, 'proposal_state'))

        chain_end_point = {key: result[:, -1] for key, result in results.items()}
        self._write_volumes(roi_indices, chain_end_point, os.path.join(self._tmp_storage_dir, 'chain_end_point'))

        if self._store_samples:
            self._write_sample_results(results, self._problem_data.mask, roi_indices)
            return results

        return SamplingProcessingWorker.SampleChainNotStored()

    def combine(self):
        super(SamplingProcessingWorker, self).combine()

        if self._store_volume_maps:
            self._combine_volumes(self._output_dir, self._tmp_storage_dir,
                                  self._problem_data.volume_header, maps_subdir='volume_maps')

        for subdir in ['proposal_state', 'chain_end_point']:
            self._combine_volumes(self._output_dir, self._tmp_storage_dir,
                                  self._problem_data.volume_header, maps_subdir=subdir)

        if self._store_samples:
            for samples in glob.glob(os.path.join(self._tmp_storage_dir, '*.samples.npy')):
                shutil.move(samples, self._output_dir)
            return load_samples(self._output_dir)

        return SamplingProcessingWorker.SampleChainNotStored()

    def _write_sample_results(self, results, full_mask, roi_indices):
        """Write the sample results to a .npy file.

        If the given sample files do not exists or if the existing file is not large enough it will create one
        with enough storage to hold all the samples for the given total_nmr_voxels.
        On storing it should also be given a list of voxel indices with the indices of the voxels that are being stored.

        Args:
            results (dict): the samples to write
            full_mask (ndarray): the complete mask for the entire brain
            roi_indices (ndarray): the roi indices of the voxels we computed
        """
        total_nmr_voxels = np.count_nonzero(full_mask)

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        for fname in os.listdir(self._output_dir):
            if fname.endswith('.samples.npy'):
                chain_name = fname[0:-len('.samples.npy')]
                if chain_name not in results:
                    os.remove(os.path.join(self._output_dir, fname))

        for map_name, samples in results.items():
            samples_path = os.path.join(self._output_dir, map_name + '.samples.npy')
            mode = 'w+'
            if os.path.isfile(samples_path):
                mode = 'r+'
                current_results = open_memmap(samples_path, mode='r')
                if current_results.shape[1] != samples.shape[1]:
                    mode = 'w+'
                del current_results # closes the memmap

            saved = open_memmap(samples_path, mode=mode, dtype=samples.dtype,
                                shape=(total_nmr_voxels, samples.shape[1]))
            saved[roi_indices, :] = samples
