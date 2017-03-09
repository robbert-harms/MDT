from contextlib import contextmanager
import logging
import os
import timeit
import time

from numpy.lib.format import open_memmap
import shutil
from mdt.nifti import load_nifti, write_all_as_nifti
from mdt.utils import model_output_exists, load_samples, restore_volumes
from mdt.processing_strategies import SimpleModelProcessingWorkerGenerator, SamplingProcessingWorker
from mdt.exceptions import InsufficientProtocolError
from mot.cl_routines.mapping.error_measures import ErrorMeasures
from mot.cl_routines.mapping.residual_calculator import ResidualCalculator

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def sample_composite_model(model, problem_data, output_folder, sampler, processing_strategy,
                           recalculate=False, store_samples=True, store_volume_maps=True):
    """Sample a composite model.

    Args:
        model (:class:`~mdt.models.composite.DMRICompositeModel`): a composite model to sample
        problem_data (:class:`~mdt.utils.DMRIProblemData`): The problem data object with which the model
            is initialized before running
        output_folder (string): The full path to the folder where to place the output
        sampler (:class:`mot.cl_routines.sampling.base.AbstractSampler`): The sampling routine to use.
        processing_strategy (:class:`~mdt.processing_strategies.ModelProcessingStrategy`): the processing strategy
        recalculate (boolean): If we want to recalculate the results if they are already present.
        store_samples (boolean): if set to False we will store none of the samples. Use this
            if you are only interested in the volume maps and not in the entire sample chain.
        store_volume_maps (boolean): if set to False we will not store the mean and std. volume maps.
    """
    if not model.is_protocol_sufficient(problem_data.protocol):
        raise InsufficientProtocolError(
            'The provided protocol is insufficient for this model. '
            'The reported errors where: {}'.format(model.get_protocol_problems(problem_data.protocol)))

    if not store_samples and not store_volume_maps:
        raise ValueError('Both store_samples and store_volume_maps are set to False, nothing to compute.')

    logger = logging.getLogger(__name__)

    if not recalculate:
        if model_output_exists(model, output_folder + '/volume_maps/', append_model_name_to_path=False):
            logger.info('Not recalculating {} model'.format(model.name))
            return load_samples(output_folder)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    model.set_problem_data(problem_data)

    with _log_info(logger, model.name):
        worker_generator = SimpleModelProcessingWorkerGenerator(
            lambda *args: SamplingProcessingWorker(sampler, store_samples, store_volume_maps, *args))
        return processing_strategy.run(model, problem_data, output_folder, recalculate, worker_generator)


@contextmanager
def _log_info(logger, model_name):
    minimize_start_time = timeit.default_timer()
    logger.info('Sampling {} model'.format(model_name))
    yield
    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Sampled {0} model with runtime {1} (h:m:s).'.format(model_name, run_time_str))


def combine_sampling_information(base_dir, append_dir, model, output_dir=None):
    """Combine the samples and sampling results from two sampling runs.

    Args:
        base_dir (str): the directory containing the first part of the total chain of samples.
        append_dir (str): the directory containing the additional samples we would like to append to the sampling
            information in the base_dir.
        model (mdt.models.composite.DMRICompositeModel): the composite model for which we are combining the samples.
            We need this to recreate the volume maps using all samples.
        output_dir (str): optionally, the output dir. If not set we default to the base_dir.
    """
    output_dir = output_dir or base_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def move_extra_maps():
        for directory in ['chain_end_point', 'proposal_state']:
            if os.path.exists(os.path.join(output_dir, directory)):
                shutil.rmtree(os.path.join(output_dir, directory))
            shutil.move(os.path.join(append_dir, directory), os.path.join(output_dir, directory))

    def append_samples():
        base_samples = load_samples(base_dir)
        other_samples = load_samples(append_dir)

        total_nmr_voxels = base_samples[list(base_samples.keys())[0]].shape[0]
        nmr_samples = [base_samples[list(base_samples.keys())[0]].shape[1],
                       other_samples[list(other_samples.keys())[0]].shape[1]]

        tmp_output_dir = os.path.join(output_dir, '_tmp_samples')
        if not os.path.exists(tmp_output_dir):
            os.makedirs(tmp_output_dir)

        for map_name in base_samples:
            samples_file = os.path.join(tmp_output_dir, map_name + '.samples.npy')

            saved = open_memmap(samples_file, mode='w+', dtype=base_samples[map_name].dtype,
                                shape=(total_nmr_voxels, sum(nmr_samples)))
            saved[:, :nmr_samples[0]] = base_samples[map_name]
            saved[:, nmr_samples[0]:] = other_samples[map_name]

            shutil.move(os.path.join(tmp_output_dir, map_name + '.samples.npy'),
                        os.path.join(output_dir, map_name + '.samples.npy'))

        shutil.rmtree(tmp_output_dir)

    def create_volume_maps():
        samples = load_samples(output_dir)

        volume_rois = model.finalize_optimization_results(model.samples_to_statistics(samples))
        errors = ResidualCalculator().calculate(model, volume_rois)
        error_measures = ErrorMeasures().calculate(errors)
        volume_rois.update(error_measures)

        mask_nifti = load_nifti(os.path.join(output_dir, 'volume_maps', 'UsedMask.nii.gz'))

        volume_maps = restore_volumes(volume_rois, mask_nifti.get_data())

        write_all_as_nifti(volume_maps, os.path.join(output_dir, 'volume_maps'), mask_nifti.get_header(),
                           overwrite_volumes=True)

    move_extra_maps()
    append_samples()
    create_volume_maps()
