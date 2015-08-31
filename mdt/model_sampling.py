import logging
import os
import shutil
import timeit
import pickle
import time
from six import string_types
from mdt.IO import Nifti
from mdt.components_loader import get_model
from mdt.utils import create_roi, configure_per_model_logging, restore_volumes, \
    ProtocolProblemError, model_output_exists

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def sample_single_model(model, problem_data, output_folder, sampler, recalculate=False, initialize=True,
                        initialize_using=None):
    """Sample a single model.

    Please note that this function does not accept cascade models.

    This will place the output in the folder: <output_folder>/<model_name>/samples/

    Args:
        model (AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize.
        problem_data (DMRIProblemData): The problem data object with which the model is initialized before running
        output_folder (string): The full path to the folder where to place the output
        sampler (AbstractSampler): The sampling routine to use.
        recalculate (boolean): If we want to recalculate the results if they are already present.
        initialize (boolean): If we want to initialize the sampler with optimization output.
            This assumes that the optimization results are in the folder:
                <output_folder>/<model_name>/
        initialize_using (None, str, or dict): If None, and initialize is True we will initialize from the
            optimization maps from a model with the same name. If a string is given and initialize is True we will
            interpret the string as a folder with the maps to load. If a dict is given and initialize is True we will
            initialize from the dict directly.
    """
    if isinstance(model, string_types):
        model = get_model(model)

    output_path = os.path.join(output_folder, model.name, 'samples')
    logger = logging.getLogger(__name__)
    model.set_problem_data(problem_data)

    if not model.is_protocol_sufficient(problem_data.protocol):
        raise ProtocolProblemError(
            'The given protocol is insufficient for this model. '
            'The reported errors where: {}'.format(model.get_protocol_problems(problem_data.protocol)))

    if recalculate:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
    else:
        if model_output_exists(model, output_folder, check_sample_output=True):
            logger.info('Not recalculating {} model'.format(model.name))
            with open(os.path.join(output_path, 'samples.pyobj'), 'rb') as f:
                return pickle.load(f)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    configure_per_model_logging(output_path)

    minimize_start_time = timeit.default_timer()
    logger.info('Sampling {} model'.format(model.name))

    init_params = _get_initialization_params(model, problem_data, output_folder, initialize, initialize_using)

    results, other_output = sampler.sample(model, init_params=init_params, full_output=True)

    _write_output(results, other_output, problem_data, output_path)

    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Sampled {0} model with runtime {1} (h:m:s).'.format(model.name, run_time_str))
    configure_per_model_logging(None)

    return results


def _write_output(results, other_output, problem_data, output_path):
    with open(os.path.join(output_path, 'samples.pyobj'), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    volume_maps_dir = os.path.join(output_path, 'volume_maps')
    if not os.path.isdir(volume_maps_dir):
        os.makedirs(volume_maps_dir)

    for key, value in other_output.items():
        if key == 'volume_maps':
            volume_maps = restore_volumes(value, problem_data.mask)
            Nifti.write_volume_maps(volume_maps, volume_maps_dir, problem_data.volume_header)
        else:
            with open(os.path.join(output_path, key + '.pyobj'), 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def _get_initialization_params(model, problem_data, output_folder, initialize, initialize_using):
    if initialize:
        maps = None
        if initialize_using is None:
            maps = Nifti.read_volume_maps(os.path.join(output_folder, model.name))
        elif isinstance(initialize_using, string_types):
            maps = Nifti.read_volume_maps(initialize_using)
        elif isinstance(initialize_using, dict):
            maps = initialize_using

        if not maps:
            raise RuntimeError('No initialization maps found in the folder "{}"'.format(
                os.path.join(output_folder, model.name)))

        init_params = create_roi(maps, problem_data.mask)
    else:
        init_params = {}

    return init_params
