import collections
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
from mdt.configuration import get_processing_strategy, get_optimizer_for_model
from mdt.models.cascade import DMRICascadeModelInterface
from mdt.utils import create_roi, get_cl_devices, model_output_exists, \
    per_model_logging_context, get_temporary_results_dir, SimpleInitializationData
from mdt.lib.processing_strategies import FittingProcessor, get_full_tmp_results_path
from mdt.lib.exceptions import InsufficientProtocolError
from mot.lib.load_balance_strategies import EvenDistribution
import mot.configuration
from mot.configuration import RuntimeConfigurationAction, CLRuntimeInfo

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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
                    logger.info('Going to fit model {0} on subject {1}'.format(model, subject_info.subject_id))

                    try:
                        model_fit = ModelFit(model,
                                             input_data,
                                             output_dir,
                                             recalculate=recalculate,
                                             only_recalculate_last=True,
                                             cl_device_ind=cl_device_ind,
                                             double_precision=double_precision,
                                             tmp_results_dir=tmp_results_dir)
                        model_fit.run()
                    except InsufficientProtocolError as ex:
                        logger.info('Could not fit model {0} on subject {1} '
                                    'due to protocol problems. {2}'.format(model, subject_info.subject_id, ex))
                    else:
                        logger.info('Done fitting model {0} on subject {1}'.format(model, subject_info.subject_id))

    return FitFunc()


class ModelFit:

    def __init__(self, model, input_data, output_folder,
                 method=None, optimizer_options=None, recalculate=False, only_recalculate_last=False,
                 cl_device_ind=None, double_precision=False, tmp_results_dir=True, initialization_data=None,
                 post_processing=None):
        """Setup model fitting for the given input model and data.

        To actually fit the model call run().

        Args:
            model (str or :class:`~mdt.models.composite.DMRICompositeModel` or :class:`~mdt.models.cascade.DMRICascadeModelInterface`):
                    the model we want to optimize.
            input_data (:class:`~mdt.utils.MRIInputData`): the input data object containing
                all the info needed for the model fitting.
            output_folder (string): The full path to the folder where to place the output
            method (str): The optimization method to use, one of:
                - 'Levenberg-Marquardt'
                - 'Nelder-Mead'
                - 'Powell'
                - 'Subplex'

                If not given, defaults to 'Powell'.
            optimizer_options (dict): extra options passed to the optimization routines.
            recalculate (boolean): If we want to recalculate the results if they are already present.
            only_recalculate_last (boolean): If we want to recalculate all the models.
                This is only of importance when dealing with CascadeModels. If set to true we only recalculate
                the last element in the chain (if recalculate is set to True, that is). If set to false,
                we recalculate everything. This only holds for the first level of the cascade.
            cl_device_ind (int or list): the index of the CL device to use. The index is from the list from the function
                get_cl_devices(). This can also be a list of device indices.
            double_precision (boolean): if we would like to do the calculations in double precision
            tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.
            initialization_data (:class:`~mdt.utils.InitializationData`): extra initialization data to use
                during model fitting. If we are optimizing a cascade model this data only applies to the last model in the
                cascade.
            post_processing (dict): a dictionary with flags for post-processing options to enable or disable.
                For valid elements, please see the configuration file settings for ``optimization``
                under ``post_processing``. Valid input for this parameter is for example: {'covariance': False}
                to disable automatic calculation of the covariance from the Hessian.

        """
        if isinstance(model, str):
            model = get_model(model)()

        if post_processing:
            model.update_active_post_processing('optimization', post_processing)

        self._model = model
        self._input_data = input_data
        self._output_folder = output_folder
        self._method = method
        self._optimizer_options = optimizer_options
        self._recalculate = recalculate
        self._only_recalculate_last = only_recalculate_last
        self._logger = logging.getLogger(__name__)

        self._model_names_list = []
        self._tmp_results_dir = get_temporary_results_dir(tmp_results_dir)
        self._initialization_data = initialization_data or SimpleInitializationData()

        if cl_device_ind is not None and not isinstance(cl_device_ind, collections.Iterable):
            cl_device_ind = [cl_device_ind]

        if cl_device_ind is not None:
            cl_environments = [get_cl_devices()[ind] for ind in cl_device_ind]
            self._cl_runtime_info = CLRuntimeInfo(cl_environments=cl_environments,
                                                  load_balancer=EvenDistribution(),
                                                  double_precision=double_precision)
        else:
            self._cl_runtime_info = CLRuntimeInfo(double_precision=double_precision)

        if not model.is_input_data_sufficient(self._input_data):
            raise InsufficientProtocolError(
                'The provided protocol is insufficient for this model. '
                'The reported errors where: {}'.format(self._model.get_input_data_problems(self._input_data)))

    def run(self):
        """Run the model and return the resulting voxel estimates within the ROI.

        Returns:
            dict: The result maps for the given composite model or the last model in the cascade.
                This returns the results as 3d/4d volumes for every output map.
        """
        _, maps = self._run(self._model, self._recalculate, self._only_recalculate_last)
        return maps

    def _run(self, model, recalculate, only_recalculate_last, _in_recursion=False):
        """Recursively calculate the (cascade) models

        Args:
            model: The model to fit, if cascade we recurse
            recalculate (boolean): if we recalculate
            only_recalculate_last: if we recalculate, if we only recalculate the last item in the first cascade
            _in_recursion (boolean): private flag, not to be set by the calling function.

        Returns:
            tuple: the first element are a dictionary with the ROI results for the maps, the second element is the
                dictionary with the reconstructed map results.
        """
        self._model_names_list.append(model.name)

        if isinstance(model, DMRICascadeModelInterface):
            all_previous_results = []
            last_results = None
            while model.has_next():
                sub_model = model.get_next(all_previous_results)

                sub_recalculate = False
                if recalculate:
                    if only_recalculate_last:
                        if not model.has_next():
                            sub_recalculate = True
                    else:
                        sub_recalculate = True

                new_in_recursion = True
                if not _in_recursion and not model.has_next():
                    new_in_recursion = False

                new_results_roi, new_results_maps = self._run(sub_model, sub_recalculate, recalculate,
                                                              _in_recursion=new_in_recursion)
                all_previous_results.append(new_results_roi)
                last_results = new_results_roi, new_results_maps
                self._model_names_list.pop()

            model.reset()
            return last_results

        return self._run_composite_model(model, recalculate, self._model_names_list,
                                         apply_user_provided_initialization=not _in_recursion)

    def _run_composite_model(self, model, recalculate, model_names, apply_user_provided_initialization=False):
        with mot.configuration.config_context(RuntimeConfigurationAction(
                cl_environments=self._cl_runtime_info.cl_environments,
                load_balancer=self._cl_runtime_info.load_balancer)):
            if apply_user_provided_initialization:
                self._apply_user_provided_initialization_data(model)

            method = self._method or get_optimizer_for_model(model_names)

            results = fit_composite_model(model, self._input_data, self._output_folder, method,
                                          self._tmp_results_dir, recalculate=recalculate, cascade_names=model_names,
                                          optimizer_options=self._optimizer_options)

        map_results = get_all_nifti_data(os.path.join(self._output_folder, model.name))
        return results, map_results

    def _apply_user_provided_initialization_data(self, model):
        """Apply the initialization data to the model.

        This has the ability to initialize maps as well as fix maps.

        Args:
            model: the composite model we are preparing for fitting. Changes happen in place.
        """
        self._logger.info('Preparing model {0} with the user provided initialization data.'.format(model.name))
        self._initialization_data.apply_to_model(model, self._input_data)


def fit_composite_model(model, input_data, output_folder, method, tmp_results_dir,
                        recalculate=False, cascade_names=None, optimizer_options=None):
    """Fits the composite model and returns the results as ROI lists per map.

     Args:
        model (:class:`~mdt.models.composite.DMRICompositeModel`): An implementation of an composite model
            that contains the model we want to optimize.
        input_data (:class:`~mdt.utils.MRIInputData`): The input data object for the model.
        output_folder (string): The path to the folder where to place the output.
            The resulting maps are placed in a subdirectory (named after the model name) in this output folder.
        method (str): The optimization routine to use.
        tmp_results_dir (str): the main directory to use for the temporary results
        recalculate (boolean): If we want to recalculate the results if they are already present.
        cascade_names (list): the list of cascade names, meant for logging
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
        logger.info('Current cascade: {0}'.format(cascade_names))

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
    minimize_start_time = timeit.default_timer()
    logger.info('Fitting {} model'.format(model_name))
    logger.info('The {} parameters we will fit are: {}'.format(len(free_param_names), free_param_names))
    yield
    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Fitted {0} model with runtime {1} (h:m:s).'.format(model_name, run_time_str))
