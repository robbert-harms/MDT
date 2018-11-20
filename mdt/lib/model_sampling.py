from contextlib import contextmanager
import logging
import os
import timeit
import time

from mdt import get_processing_strategy
from mdt.utils import load_samples, per_model_logging_context
from mdt.lib.processing_strategies import SamplingProcessor, SaveAllSamples, \
    SaveNoSamples, get_full_tmp_results_path, SaveSpecificMaps
from mdt.lib.exceptions import InsufficientProtocolError


__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def sample_composite_model(model, input_data, output_folder, nmr_samples, thinning, burnin, tmp_dir,
                           method=None, recalculate=False, store_samples=True, sample_items_to_save=None,
                           initialization_data=None, post_sampling_cb=None, sampler_options=None):
    """Sample a composite model.

    Args:
        model (:class:`~mdt.models.composite.DMRICompositeModel`): a composite model to sample
        input_data (:class:`~mdt.utils.MRIInputData`): The input data object with which the model
            is initialized before running
        output_folder (string): The full path to the folder where to place the output
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
        initialization_data (:class:`~mdt.utils.InitializationData`): provides (extra) initialization data to use
            during model fitting. If we are optimizing a cascade model this data only applies to the last model in the
            cascade.
        post_sampling_cb (Callable[
            [mot.sample.base.SamplingOutput, mdt.models.composite.DMRICompositeModel], Optional[Dict]]):
                additional post-processing called after sampling. This function can optionally return a (nested)
                dictionary with as keys dir-/file-names and as values maps to be stored in the results directory.
        sampler_options (dict): specific options for the MCMC routine. These will be provided to the sampling routine
            as additional keyword arguments to the constructor.
    """
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

    logger = logging.getLogger(__name__)

    if not recalculate:
        if os.path.exists(os.path.join(output_folder, 'UsedMask.nii.gz')) \
                or os.path.exists(os.path.join(output_folder, 'UsedMask.nii')):
            logger.info('Not recalculating {} model'.format(model.name))
            return load_samples(output_folder)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    model.set_input_data(input_data)

    with per_model_logging_context(output_folder, overwrite=recalculate):
        if initialization_data:
            logger.info('Preparing the model with the user provided initialization data.')
            initialization_data.apply_to_model(model, input_data)

        with _log_info(logger, model.name):
            worker = SamplingProcessor(
                nmr_samples, thinning, burnin, method or 'AMWG',
                model, input_data.mask, input_data.nifti_header, output_folder,
                get_full_tmp_results_path(output_folder, tmp_dir), recalculate,
                samples_storage_strategy=samples_storage_strategy,
                post_sampling_cb=post_sampling_cb,
                sampler_options=sampler_options)

            processing_strategy = get_processing_strategy('sampling')
            return processing_strategy.process(worker)


@contextmanager
def _log_info(logger, model_name):
    minimize_start_time = timeit.default_timer()
    logger.info('Sampling {} model'.format(model_name))
    yield
    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Sampled {0} model with runtime {1} (h:m:s).'.format(model_name, run_time_str))
