import numpy as np
import glob
import logging
import os
import shutil
import time
import timeit
from contextlib import contextmanager
from mdt.__version__ import __version__
from mdt.lib.nifti import get_all_nifti_data
from mdt.lib.components import get_model
from mdt.configuration import get_processing_strategy
from mdt.utils import create_roi, get_cl_devices, model_output_exists, \
    per_model_logging_context
from mdt.lib.processing_strategies import FittingProcessor, get_full_tmp_results_path
from mdt.lib.exceptions import InsufficientProtocolError
import mot.configuration
from mot.configuration import config_context as mot_config_context

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_optimization_inits(model_name, input_data, output_folder, cl_device_ind=None):
    """Get better optimization starting points for the given model.

    Since initialization can make quite a difference in optimization results, this function can generate
    a good initialization starting point for the given model. The idea is that before you call the :func:`fit_model`
    function, you call this function to get a better starting point. An usage example would be::

        input_data = mdt.load_input_data(..)

        init_data = get_optimization_inits('BallStick_r1', input_data, '/my/folder')

        fit_model('BallStick_r1', input_data, '/my/folder',
                  initialization_data={'inits': init_data})

    Where the init data returned by this function can directly be used as input to the ``initialization_data``
    argument of the :func`fit_model` function.

    Please note that his function only supports models shipped by default with MDT.

    Args:
        model_name (str):
            The name of a model for which we want the optimization starting points.
        input_data (:class:`~mdt.utils.MRIInputData`): the input data object containing all
            the info needed for model fitting of intermediate models.
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it.
        cl_device_ind (int or list): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices(). This can also be a list of device indices.

    Returns:
        dict: a dictionary with initialization points for the selected model
    """
    logger = logging.getLogger(__name__)

    def get_subset(param_names, fit_results):
        return {key: value for key, value in fit_results.items() if key in param_names}

    def get_model_fit(model_name):
        logger.info('Starting intermediate optimization for generating initialization point.')

        from mdt import fit_model
        results = fit_model(model_name, input_data, output_folder, recalculate=False, use_cascaded_inits=False,
                            initialization_data={'inits': get_init_data(model_name)})

        logger.info('Finished intermediate optimization for generating initialization point.')
        return results

    def get_init_data(model_name):
        inits = {}
        free_parameters = get_model(model_name)().get_free_param_names()

        if 'S0.s0' in free_parameters and input_data.has_input_data('b'):
            unweighted_locations = np.where(input_data.get_input_data('b') < 250e6)[0]
            inits['S0.s0'] = np.mean(input_data.signal4d[..., unweighted_locations], axis=-1)

        if model_name.startswith('BallStick_r2'):
            inits.update(get_subset(free_parameters, get_model_fit('BallStick_r1')))
            inits['w_stick1.w'] = np.minimum(inits['w_stick0.w'], 0.05)
        elif model_name.startswith('BallStick_r3'):
            inits.update(get_subset(free_parameters, get_model_fit('BallStick_r2')))
            inits['w_stick2.w'] = np.minimum(inits['w_stick1.w'], 0.05)
        elif model_name.startswith('Tensor'):
            fit_results = get_model_fit('BallStick_r1')
            inits.update(get_subset(free_parameters, fit_results))
            inits['Tensor.theta'] = fit_results['Stick0.theta']
            inits['Tensor.phi'] = fit_results['Stick0.phi']
        elif model_name.startswith('NODDI'):
            fit_results = get_model_fit('BallStick_r1')
            inits.update(get_subset(free_parameters, fit_results))
            inits['w_ic.w'] = fit_results['w_stick0.w'] / 2.0
            inits['w_ec.w'] = fit_results['w_stick0.w'] / 2.0
            inits['w_csf.w'] = fit_results['w_ball.w']
            inits['NODDI_IC.theta'] = fit_results['Stick0.theta']
            inits['NODDI_IC.phi'] = fit_results['Stick0.phi']
        elif model_name.startswith('BinghamNODDI_r1'):
            noddi_results = get_model_fit('NODDI')
            inits.update(get_subset(free_parameters, noddi_results))
            inits['w_in0.w'] = noddi_results['w_ic.w']
            inits['w_en0.w'] = noddi_results['w_ec.w']
            inits['w_csf.w'] = noddi_results['w_csf.w']
            inits['BinghamNODDI_IN0.theta'] = noddi_results['NODDI_IC.theta']
            inits['BinghamNODDI_IN0.phi'] = noddi_results['NODDI_IC.phi']
            inits['BinghamNODDI_IN0.k1'] = noddi_results['NODDI_IC.kappa']
        elif model_name.startswith('BinghamNODDI_r2'):
            bs2_results = get_model_fit('BallStick_r2')
            inits.update(get_subset(free_parameters, bs2_results))
            inits.update(get_subset(free_parameters, get_model_fit('BinghamNODDI_r1')))
            inits['BinghamNODDI_IN1.theta'] = bs2_results['Stick1.theta']
            inits['BinghamNODDI_IN1.phi'] = bs2_results['Stick1.phi']
        elif model_name.startswith('Kurtosis'):
            fit_results = get_model_fit('Tensor')
            inits.update(get_subset(free_parameters, fit_results))
            inits.update({'KurtosisTensor.' + key: fit_results['Tensor.' + key]
                          for key in ['theta', 'phi', 'psi', 'd', 'dperp0', 'dperp1']})
        elif model_name.startswith('CHARMED_r'):
            nmr_dir = model_name[len('CHARMED_r'):len('CHARMED_r')+1]
            fit_results = get_model_fit('BallStick_r' + nmr_dir)
            inits.update(get_subset(free_parameters, fit_results))
            inits['Tensor.theta'] = fit_results['Stick0.theta']
            inits['Tensor.phi'] = fit_results['Stick0.phi']
            for dir_ind in range(int(nmr_dir)):
                inits['w_res{}.w'.format(dir_ind)] = fit_results['w_stick{}.w'.format(dir_ind)]
                inits['CHARMEDRestricted{}.theta'.format(dir_ind)] = fit_results['Stick{}.theta'.format(dir_ind)]
                inits['CHARMEDRestricted{}.phi'.format(dir_ind)] = fit_results['Stick{}.phi'.format(dir_ind)]
        elif model_name.startswith('BallRacket_r'):
            nmr_dir = model_name[len('BallRacket_r'):len('BallRacket_r')+1]
            fit_results = get_model_fit('BallStick_r' + nmr_dir)
            inits.update(get_subset(free_parameters, fit_results))
            for dir_ind in range(int(nmr_dir)):
                inits['w_res{}.w'.format(dir_ind)] = fit_results['w_stick{}.w'.format(dir_ind)]
                inits['Racket{}.theta'.format(dir_ind)] = fit_results['Stick{}.theta'.format(dir_ind)]
                inits['Racket{}.phi'.format(dir_ind)] = fit_results['Stick{}.phi'.format(dir_ind)]
        elif model_name.startswith('AxCaliber'):
            fit_results = get_model_fit('BallStick_r1')
            inits.update(get_subset(free_parameters, fit_results))
            inits['GDRCylinders.theta'] = fit_results['Stick0.theta']
            inits['GDRCylinders.phi'] = fit_results['Stick0.phi']
        elif model_name.startswith('ActiveAx'):
            fit_results = get_model_fit('BallStick_r1')
            inits.update(get_subset(free_parameters, fit_results))
            inits['w_ic.w'] = fit_results['w_stick0.w'] / 2.0
            inits['w_ec.w'] = fit_results['w_stick0.w'] / 2.0
            inits['w_csf.w'] = fit_results['w_ball.w']
            inits['CylinderGPD.theta'] = fit_results['Stick0.theta']
            inits['CylinderGPD.phi'] = fit_results['Stick0.phi']
        elif model_name.startswith('QMT_ReducedRamani'):
            inits['S0.s0'] = np.mean(input_data.signal4d, axis=-1)

        return inits

    cl_environments = None
    if cl_device_ind is not None:
        cl_environments = get_cl_devices(cl_device_ind)

    with mot_config_context(mot.configuration.RuntimeConfigurationAction(cl_environments=cl_environments)):
        return get_init_data(model_name)


def get_batch_fitting_function(total_nmr_subjects, models_to_fit, output_folder,
                               recalculate=False, cl_device_ind=None, double_precision=False,
                               tmp_results_dir=True, use_gradient_deviations=False):
    """Get the batch fitting function that can fit all desired models on a subject.

    Args:
        total_nmr_subjects (int): the total number of subjects we are fitting.
        models_to_fit (list of str): A list of models to fit to the data.
        output_folder (str): the folder in which to place the output
        recalculate (boolean): If we want to recalculate the results if they are already present.
        cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
            get_cl_devices().
        double_precision (boolean): if we would like to do the calculations in double precision
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
            that path directly, set to True to use the config value, set to None to disable.
        use_gradient_deviations (boolean): if you want to use the gradient deviations if present
    """
    logger = logging.getLogger(__name__)

    @contextmanager
    def timer(subject_id):
        start_time = timeit.default_timer()
        yield
        logger.info('Fitted all models on subject {0} in time {1} (h:m:s)'.format(
            subject_id, time.strftime('%H:%M:%S', time.gmtime(timeit.default_timer() - start_time))))

    class FitFunc:

        def __init__(self):
            self._index_counter = 0

        def __call__(self, subject_info):
            from mdt import fit_model

            logger.info('Going to process subject {}, ({} of {}, we are at {:.2%})'.format(
                subject_info.subject_id, self._index_counter + 1, total_nmr_subjects,
                self._index_counter / total_nmr_subjects))
            self._index_counter += 1

            output_dir = os.path.join(output_folder, subject_info.subject_id)

            if all(model_output_exists(model, output_dir) for model in models_to_fit) and not recalculate:
                logger.info('Skipping subject {0}, output exists'.format(subject_info.subject_id))
                return

            logger.info('Loading the data (DWI, mask and protocol) of subject {0}'.format(subject_info.subject_id))
            input_data = subject_info.get_input_data(use_gradient_deviations)

            with timer(subject_info.subject_id):
                for model in models_to_fit:
                    if isinstance(model, str):
                        model_name = model
                    else:
                        model_name = model.name

                    logger.info('Going to fit model {0} on subject {1}'.format(model_name, subject_info.subject_id))

                    try:
                        fit_model(model, input_data, output_dir,
                                  recalculate=recalculate,
                                  cl_device_ind=cl_device_ind,
                                  double_precision=double_precision,
                                  tmp_results_dir=tmp_results_dir,
                                  use_cascaded_inits=True)

                    except InsufficientProtocolError as ex:
                        logger.info('Could not fit model {0} on subject {1} '
                                    'due to protocol problems. {2}'.format(model_name, subject_info.subject_id, ex))
                    else:
                        logger.info('Done fitting model {0} on subject {1}'.format(model_name, subject_info.subject_id))

    return FitFunc()


def fit_composite_model(model, input_data, output_folder, method, tmp_results_dir,
                        recalculate=False, optimizer_options=None):
    """Fits the composite model and returns the results as ROI lists per map.

     Args:
        model (:class:`~mdt.models.base.EstimableModel`): An implementation of an composite model
            that contains the model we want to optimize.
        input_data (:class:`~mdt.utils.MRIInputData`): The input data object for the model.
        output_folder (string): The path to the folder where to place the output.
            The resulting maps are placed in a subdirectory (named after the model name) in this output folder.
        method (str): The optimization routine to use.
        tmp_results_dir (str): the main directory to use for the temporary results
        recalculate (boolean): If we want to recalculate the results if they are already present.
        optimizer_options (dict): the additional optimization options
    """
    logger = logging.getLogger(__name__)
    output_path = os.path.join(output_folder, model.name)

    if not model.is_input_data_sufficient(input_data):
        raise InsufficientProtocolError(
            'The given protocol is insufficient for this model. '
            'The reported errors where: {}'.format(model.get_input_data_problems(input_data)))

    if not recalculate and model_output_exists(model, output_folder):
        maps = get_all_nifti_data(output_path)
        logger.info('Not recalculating {} model'.format(model.name))
        return create_roi(maps, input_data.mask)

    with per_model_logging_context(output_path):
        logger.info('Using MDT version {}'.format(__version__))
        logger.info('Preparing for model {0}'.format(model.name))

        model.set_input_data(input_data)

        if recalculate:
            if os.path.exists(output_path):
                list(map(os.remove, glob.glob(os.path.join(output_path, '*.nii*'))))
                if os.path.exists(os.path.join(output_path + 'covariances')):
                    shutil.rmtree(os.path.join(output_path + 'covariances'))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with _model_fit_logging(logger, model.name, model.get_free_param_names()):
            tmp_dir = get_full_tmp_results_path(output_path, tmp_results_dir)
            logger.info('Saving temporary results in {}.'.format(tmp_dir))

            worker = FittingProcessor(method, model, input_data.mask,
                                      input_data.nifti_header, output_path,
                                      tmp_dir, recalculate, optimizer_options=optimizer_options)

            processing_strategy = get_processing_strategy('optimization')
            return processing_strategy.process(worker)


@contextmanager
def _model_fit_logging(logger, model_name, free_param_names):
    """Adds logging information around the processing."""
    def calculate_run_days(runtime):
        if runtime > 24 * 60 * 60:
            return int(runtime // (24. * 60 * 60))
        return 0

    minimize_start_time = timeit.default_timer()
    logger.info('Fitting {} model'.format(model_name))
    logger.info('The {} parameters we will fit are: {}'.format(len(free_param_names), free_param_names))
    yield
    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = str(calculate_run_days(run_time)) + ':' + time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Fitted {0} model with runtime {1} (d:h:m:s).'.format(model_name, run_time_str))
