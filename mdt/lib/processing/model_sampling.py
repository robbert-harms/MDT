import collections
import shutil
from contextlib import contextmanager
import logging
import os
import timeit
import time
import numpy as np
from numpy.lib.format import open_memmap
from mdt.configuration import gzip_sampling_results, get_processing_strategy
from mdt.lib.fsl_sampling_routine import FSLSamplingRoutine
from mdt.utils import load_samples, per_model_logging_context, get_intermediate_results_path
from mdt.lib.processing.processing_strategies import SimpleModelProcessor
from mdt.lib.exceptions import InsufficientProtocolError
from mot.sample import AdaptiveMetropolisWithinGibbs, SingleComponentAdaptiveMetropolis, MetropolisWithinGibbs
from mot.sample.t_walk import ThoughtfulWalk

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert@xkls.nl"


def sample_composite_model(model, input_data, output_folder, nmr_samples, thinning, burnin, tmp_dir,
                           method=None, recalculate=False, store_samples=True, sample_items_to_save=None,
                           post_sampling_cb=None, sampler_options=None):
    """Sample a composite model.

    Args:
        model (:class:`~mdt.models.base.EstimableModel`): a composite model to sample
        input_data (:class:`~mdt.lib.input_data.MRIInputData`): The input data object with which the model
            is initialized before running
        output_folder (string): The relative output path.
            The resulting maps are placed in a subdirectory (named after the model name) in this output folder.
        nmr_samples (int): the number of samples we would like to return.
        burnin (int): the number of samples to burn-in, that is, to discard before returning the desired
            number of samples
        thinning (int): how many sample we wait before storing a new one. This will draw extra samples such that
                the total number of samples generated is ``nmr_samples * (thinning)`` and the number of samples stored
                is ``nmr_samples``. If set to one or lower we store every sample after the burn in.
        tmp_dir (str): the preferred temporary storage dir
        method (str): The sampling method to use, one of:
            - 'AMWG', for the Adaptive Metropolis-Within-Gibbs method
            - 'SCAM', for the Single Component Adaptive Metropolis
            - 'FSL', for the sampling method used in the FSL toolbox
            - 'MWG', for the Metropolis-Within-Gibbs (simple random walk metropolis without updates)

            If not given, defaults to 'AMWG'.
        recalculate (boolean): If we want to recalculate the results if they are already present.
        store_samples (boolean, sequence or :class:`mdt.lib.processing_strategies.SamplesStorageStrategy`): if set to
            False, we will store none of the samples. If set to True we will save all samples. If set to a sequence we
            expect a sequence of integer numbers with sample positions to store. Finally, you can also give a subclass
            instance of :class:`~mdt.lib.processing_strategies.SamplesStorageStrategy` (it is then typically set to
            a :class:`mdt.lib.processing_strategies.SaveThinnedSamples` instance).
        sample_items_to_save (list): list of output names we want to store the samples of. If given, we only
            store the items specified in this list. Valid items are the free parameter names of the model and the
            items 'LogLikelihood' and 'LogPrior'.
        post_sampling_cb (Callable[
            [mot.sample.base.SamplingOutput, mdt.models.base.EstimableModel], Optional[Dict]]):
                additional post-processing called after sampling. This function can optionally return a (nested)
                dictionary with as keys dir-/file-names and as values maps to be stored in the results directory.
        sampler_options (dict): specific options for the MCMC routine. These will be provided to the sampling routine
            as additional keyword arguments to the constructor.
    """
    from mdt.__version__ import __version__
    logger = logging.getLogger(__name__)
    logger.info('Using MDT version {}'.format(__version__))
    logger.info('Preparing for model {0}'.format(model.name))
    logger.info('The {0} parameters we will sample are: {1}'.format(len(model.get_free_param_names()),
                                                                        model.get_free_param_names()))

    output_folder = os.path.join(output_folder, model.name, 'samples')

    samples_storage_strategy = SaveAllSamples()
    if store_samples:
        if sample_items_to_save:
            samples_storage_strategy = SaveSpecificMaps(included=sample_items_to_save)
    else:
        samples_storage_strategy = SaveNoSamples()

    if not model.is_input_data_sufficient(input_data):
        raise InsufficientProtocolError(
            'The provided protocol is insufficient for this model. '
            'The reported errors where: {}'.format(model.get_input_data_problems(input_data)))

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if recalculate:
        shutil.rmtree(output_folder)
    else:
        if os.path.exists(os.path.join(output_folder, 'UsedMask.nii.gz')) \
                or os.path.exists(os.path.join(output_folder, 'UsedMask.nii')):
            logger.info('Not recalculating {} model'.format(model.name))
            return load_samples(output_folder)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    model.set_input_data(input_data)

    with per_model_logging_context(output_folder, overwrite=recalculate):
        with _log_info(logger, model.name):
            worker = SamplingProcessor(
                nmr_samples, thinning, burnin, method or 'AMWG',
                model, input_data.mask, input_data.nifti_header, output_folder,
                get_intermediate_results_path(output_folder, tmp_dir), recalculate,
                samples_storage_strategy=samples_storage_strategy,
                post_sampling_cb=post_sampling_cb,
                sampler_options=sampler_options)

            processing_strategy = get_processing_strategy('sampling')
            return processing_strategy.process(worker)


@contextmanager
def _log_info(logger, model_name):
    def calculate_run_days(runtime):
        if runtime > 24 * 60 * 60:
            return int(runtime // (24. * 60 * 60))
        return 0

    minimize_start_time = timeit.default_timer()
    logger.info('Sampling {} model'.format(model_name))
    yield
    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = str(calculate_run_days(run_time)) + ':' + time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Sampled {0} model with runtime {1} (d:h:m:s).'.format(model_name, run_time_str))


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
                [mot.sample.base.SamplingOutput, mdt.models.base.EstimableModel], Optional[Dict]]):
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

        self._kernel_data = self._model.get_kernel_data()
        self._initial_params = self._model.get_initial_parameters()
        self._ll_func = self._model.get_log_likelihood_function()
        self._prior_func = self._model.get_log_prior_function()

    def _process(self, roi_indices, next_indices=None):
        method = None
        method_args = [self._ll_func, self._prior_func, self._initial_params[roi_indices]]
        method_kwargs = {'data': self._kernel_data.get_subset(roi_indices)}

        if self._method in ['AMWG', 'SCAM', 'MWG', 'FSL']:
            method_args.append(self._model.get_rwm_proposal_stds()[roi_indices])

        if self._method in ['AMWG', 'SCAM', 'MWG', 'FSL', 't-walk']:
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
            method_args.append(self._model.get_random_parameter_positions(nmr_positions=1)[roi_indices, :, 0])

        method_kwargs.update(self._sampler_options)

        if method is None:
            raise ValueError('Could not find the sampler with name {}.'.format(self._method))

        sampler = method(*method_args, **method_kwargs)
        sampling_output = sampler.sample(self._nmr_samples, burnin=self._burnin, thinning=self._thinning)
        samples = sampling_output.get_samples()

        self._logger.info('Starting post-processing')
        maps_to_save = self._model.get_post_sampling_maps(sampling_output, roi_indices=roi_indices)
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
