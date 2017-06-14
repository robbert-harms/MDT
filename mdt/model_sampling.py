from contextlib import contextmanager
import logging
import os
import timeit
import time
from mdt.utils import model_output_exists, load_samples, restore_volumes
from mdt.processing_strategies import SimpleModelProcessingWorkerGenerator, SamplingProcessingWorker, SaveAllSamples, \
    SaveNoSamples, SaveThinnedSamples
from mdt.exceptions import InsufficientProtocolError


__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def sample_composite_model(model, problem_data, output_folder, sampler, processing_strategy,
                           recalculate=False, store_samples=True, store_volume_maps=True,
                           initialization_data=None):
    """Sample a composite model.

    Args:
        model (:class:`~mdt.models.composite.DMRICompositeModel`): a composite model to sample
        problem_data (:class:`~mdt.utils.DMRIProblemData`): The problem data object with which the model
            is initialized before running
        output_folder (string): The full path to the folder where to place the output
        sampler (:class:`mot.cl_routines.sampling.base.AbstractSampler`): The sampling routine to use.
        processing_strategy (:class:`~mdt.processing_strategies.ModelProcessingStrategy`): the processing strategy
        recalculate (boolean): If we want to recalculate the results if they are already present.
        store_samples (boolean or int): if set to False we will store none of the samples. If set to an integer we will
            store only thinned samples with that amount.
        store_volume_maps (boolean): if set to False we will not store the mean and std. volume maps.
        initialization_data (:class:`~mdt.utils.InitializationData`): provides (extra) initialization data to use
            during model fitting. If we are optimizing a cascade model this data only applies to the last model in the
            cascade.
    """
    if store_samples:
        if isinstance(store_samples, int):
            sample_to_save_method = SaveThinnedSamples(store_samples)
        else:
            sample_to_save_method = SaveAllSamples()
    else:
        sample_to_save_method = SaveNoSamples()

    if not model.is_protocol_sufficient(problem_data.protocol):
        raise InsufficientProtocolError(
            'The provided protocol is insufficient for this model. '
            'The reported errors where: {}'.format(model.get_protocol_problems(problem_data.protocol)))

    if not sample_to_save_method.store_samples() and not store_volume_maps:
        raise ValueError('Both store_samples and store_volume_maps are set to False, nothing to compute.')

    logger = logging.getLogger(__name__)

    if not recalculate:
        if model_output_exists(model, output_folder + '/volume_maps/', append_model_name_to_path=False):
            logger.info('Not recalculating {} model'.format(model.name))
            return load_samples(output_folder)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    model.set_problem_data(problem_data)

    if initialization_data:
        logger.info('Preparing the model with the user provided initialization data.')
        initialization_data.apply_to_model(model, problem_data)

    with _log_info(logger, model.name):
        worker_generator = SimpleModelProcessingWorkerGenerator(
            lambda *args: SamplingProcessingWorker(sampler, sample_to_save_method, store_volume_maps, *args))
        return processing_strategy.run(model, problem_data, output_folder, recalculate, worker_generator)


@contextmanager
def _log_info(logger, model_name):
    minimize_start_time = timeit.default_timer()
    logger.info('Sampling {} model'.format(model_name))
    yield
    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Sampled {0} model with runtime {1} (h:m:s).'.format(model_name, run_time_str))
