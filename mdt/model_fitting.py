import collections
import glob
import logging
import os
import time
import timeit
from contextlib import contextmanager
from six import string_types
from mdt.__version__ import __version__
from mdt.nifti import get_all_image_data
from mdt.batch_utils import batch_profile_factory, AllSubjects
from mdt.components_loader import get_model
from mdt.configuration import get_processing_strategy, get_optimizer_for_model
from mdt.models.cascade import DMRICascadeModelInterface
from mdt.protocols import write_protocol
from mdt.utils import create_roi, get_cl_devices, model_output_exists, \
    per_model_logging_context, get_temporary_results_dir, SimpleInitializationData, is_scalar
from mdt.processing_strategies import SimpleModelProcessingWorkerGenerator, FittingProcessingWorker
from mdt.exceptions import InsufficientProtocolError
from mot.load_balance_strategies import EvenDistribution
import mot.configuration
from mot.configuration import RuntimeConfigurationAction

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BatchFitting(object):

    def __init__(self, data_folder, models_to_fit, batch_profile=None, subjects_selection=None, recalculate=False,
                 cascade_subdir=False, cl_device_ind=None, double_precision=False, tmp_results_dir=True):
        """This class is meant to make running computations as simple as possible.

        The idea is that a single folder is enough to fit_model the computations. One can optionally give it the
        batch_profile to use for the fitting. If not given, this class will attempt to use the
        batch_profile that fits the data folder best.

        Setting the ``cl_device_ind`` has the side effect that it changes the current run time cl_device settings in the
        MOT toolkit for the duration of this function.

        Args:
            data_folder (str): the main directory to look for items to process.
            models_to_fit (list of str): A list of models to fit to the data.
            batch_profile (:class:`~mdt.batch_utils.BatchProfile` or str): the batch profile to use
                or the name of a batch profile to use from the users folder.
            subjects_selection (:class:`~mdt.batch_utils.BatchSubjectSelection`): the subjects to use for processing.
                If None all subjects are processed.
            recalculate (boolean): If we want to recalculate the results if they are already present.
            cascade_subdir (boolean): if we want to create a subdirectory for every cascade model.
                Per default we output the maps of cascaded results in the same directory, this allows reusing cascaded
                results for other cascades (for example, if you cascade BallStick -> Noddi you can use
                the BallStick results also for BallStick -> Charmed). This flag disables that behaviour and instead
                outputs the results of a cascade model to a subdirectory for that cascade.
                This does not apply recursive.
            cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
                get_cl_devices().
            double_precision (boolean): if we would like to do the calculations in double precision
            tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.
        """
        self._logger = logging.getLogger(__name__)
        self._batch_profile = batch_profile_factory(batch_profile, data_folder)
        self._subjects_selection = subjects_selection or AllSubjects()
        self._tmp_results_dir = tmp_results_dir
        self._models_to_fit = models_to_fit
        self._cl_device_ind = cl_device_ind
        self._recalculate = recalculate
        self._double_precision = double_precision
        self._cascade_subdir = cascade_subdir

        if self._batch_profile is None:
            raise RuntimeError('No suitable batch profile could be '
                               'found for the directory {0}'.format(os.path.abspath(data_folder)))

        self._logger.info('Using MDT version {}'.format(__version__))
        self._logger.info('Using batch profile: {0}'.format(self._batch_profile))
        self._subjects = self._subjects_selection.get_selection(self._batch_profile.get_subjects())

        self._logger.info('Subjects found: {0}'.format(self._batch_profile.get_subjects_count()))
        self._logger.info('Subjects to process: {0}'.format(len(self._subjects)))
        self._logger.info('Going to fit these models to all data: {}'.format(self._models_to_fit))

        if self._cl_device_ind is not None:
            if not isinstance(self._cl_device_ind, collections.Iterable):
                self._cl_device_ind = [self._cl_device_ind]
            devices = get_cl_devices()
            mot.configuration.set_cl_environments([devices[ind] for ind in self._cl_device_ind])

    def get_all_subjects_info(self):
        """Get a dictionary with the info about all the found subjects.

        This will return information about all the subjects found and will disregard the current ``subjects`` setting
        that limits the amount of subjects we will run.

        Returns:
            list of :class:`~mdt.batch_utils.SubjectInfo`: information about all available subjects
        """
        return self._batch_profile.get_subjects()

    def get_subjects_info(self):
        """Get a dictionary with the info of the subject we will run computations on.

        This will return information about the subjects that we will use in the batch fitting.

        Returns:
            list of :class:`~mdt.batch_utils.SubjectInfo`: information about all subjects we will actually use
        """
        return self._subjects

    def run(self):
        """Run the computations on the current dir with all the configured options. """
        self._logger.info('Running computations on {0} subjects'.format(len(self._subjects)))

        run_func = _BatchFitRunner(self._models_to_fit, self._recalculate, self._cascade_subdir,
                                   self._cl_device_ind, self._double_precision, self._tmp_results_dir)
        for ind, subject in enumerate(self._subjects):
            self._logger.info('Going to process subject {}, ({} of {}, we are at {:.2%})'.format(
                subject.subject_id, ind + 1, len(self._subjects), ind / len(self._subjects)))
            run_func(subject)

        return self._subjects


class _BatchFitRunner(object):

    def __init__(self, models_to_fit, recalculate, cascade_subdir, cl_device_ind, double_precision, tmp_results_dir):
        self._models_to_fit = models_to_fit
        self._recalculate = recalculate
        self._cascade_subdir = cascade_subdir
        self._cl_device_ind = cl_device_ind
        self._double_precision = double_precision
        self._logger = logging.getLogger(__name__)
        self._tmp_results_dir = tmp_results_dir

    def __call__(self, subject_info):
        """Run the batch fitting on the given subject.

        This is a module level function to allow for python multiprocessing to work.

        Args:
            subject_info (SubjectInfo): the subject information
        """
        output_dir = subject_info.output_dir

        if all(model_output_exists(model, output_dir) for model in self._models_to_fit) and not self._recalculate:
            self._logger.info('Skipping subject {0}, output exists'.format(subject_info.subject_id))
            return

        self._logger.info('Loading the data (DWI, mask and protocol) of subject {0}'.format(subject_info.subject_id))
        problem_data = subject_info.get_problem_data()

        with self._timer(subject_info.subject_id):
            for model in self._models_to_fit:
                self._logger.info('Going to fit model {0} on subject {1}'.format(model, subject_info.subject_id))
                try:
                    model_fit = ModelFit(model,
                                         problem_data,
                                         output_dir,
                                         recalculate=self._recalculate,
                                         only_recalculate_last=True,
                                         cascade_subdir=self._cascade_subdir,
                                         cl_device_ind=self._cl_device_ind,
                                         double_precision=self._double_precision,
                                         tmp_results_dir=self._tmp_results_dir)
                    model_fit.run()
                except InsufficientProtocolError as ex:
                    self._logger.info('Could not fit model {0} on subject {1} '
                                      'due to protocol problems. {2}'.format(model, subject_info.subject_id, ex))
                else:
                    self._logger.info('Done fitting model {0} on subject {1}'.format(model, subject_info.subject_id))

    @contextmanager
    def _timer(self, subject_id):
        start_time = timeit.default_timer()
        yield
        self._logger.info('Fitted all models on subject {0} in time {1} (h:m:s)'.format(
            subject_id, time.strftime('%H:%M:%S', time.gmtime(timeit.default_timer() - start_time))))


class ModelFit(object):

    def __init__(self, model, problem_data, output_folder, optimizer=None,
                 recalculate=False, only_recalculate_last=False, cascade_subdir=False,
                 cl_device_ind=None, double_precision=False, tmp_results_dir=True, initialization_data=None):
        """Setup model fitting for the given input model and data.

        To actually fit the model call run().

        Args:
            model (str or :class:`~mdt.models.composite.DMRICompositeModel` or :class:`~mdt.models.cascade.DMRICascadeModelInterface`):
                    the model we want to optimize.
            problem_data (:class:`~mdt.utils.DMRIProblemData`): the problem data object which contains the dwi image,
                the dwi header, the brain_mask and the protocol to use.
            output_folder (string): The full path to the folder where to place the output
            optimizer (:class:`mot.cl_routines.optimizing.base.AbstractOptimizer`): The optimization routine to use.
                If None, we create one using the configuration files.
            recalculate (boolean): If we want to recalculate the results if they are already present.
            only_recalculate_last (boolean): If we want to recalculate all the models.
                This is only of importance when dealing with CascadeModels. If set to true we only recalculate
                the last element in the chain (if recalculate is set to True, that is). If set to false,
                we recalculate everything. This only holds for the first level of the cascade.
            cascade_subdir (boolean): if we want to create a subdirectory for the given model if it is a cascade model.
                Per default we output the maps of cascaded results in the same directory, this allows reusing cascaded
                results for other cascades (for example, if you cascade BallStick -> Noddi you can use the BallStick
                results also for BallStick -> Charmed). This flag disables that behaviour and instead outputs the
                results of a cascade model to a subdirectory for that cascade. This does not apply recursive.
            cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
                get_cl_devices(). This can also be a list of device indices.
            double_precision (boolean): if we would like to do the calculations in double precision
            tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.
            initialization_data (:class:`~mdt.utils.InitializationData`): extra initialization data to use
                during model fitting. If we are optimizing a cascade model this data only applies to the last model in the
                cascade.
        """
        if isinstance(model, string_types):
            model = get_model(model)

        model.double_precision = double_precision

        self._model = model
        self._problem_data = problem_data
        self._output_folder = output_folder
        if cascade_subdir and isinstance(self._model, DMRICascadeModelInterface):
            self._output_folder += '/{}'.format(self._model.name)
        self._optimizer = optimizer
        self._recalculate = recalculate
        self._only_recalculate_last = only_recalculate_last
        self._logger = logging.getLogger(__name__)
        self._cl_device_indices = cl_device_ind
        self._model_names_list = []
        self._tmp_results_dir = get_temporary_results_dir(tmp_results_dir)
        self._initialization_data = initialization_data or SimpleInitializationData()

        if self._cl_device_indices is not None and not isinstance(self._cl_device_indices, collections.Iterable):
            self._cl_device_indices = [self._cl_device_indices]

        self._cl_envs = None
        self._load_balancer = None
        if self._cl_device_indices is not None:
            all_devices = get_cl_devices()
            self._cl_envs = [all_devices[ind] for ind in self._cl_device_indices]
            self._load_balancer = EvenDistribution()

        if not model.is_protocol_sufficient(self._problem_data.protocol):
            raise InsufficientProtocolError(
                'The provided protocol is insufficient for this model. '
                'The reported errors where: {}'.format(self._model.get_protocol_problems(
                    self._problem_data.protocol)))

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
            results = {}
            last_results = None
            while model.has_next():
                sub_model = model.get_next(results)

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
                results.update({sub_model.name: new_results_roi})
                last_results = new_results_roi, new_results_maps
                self._model_names_list.pop()

            model.reset()
            return last_results

        return self._run_composite_model(model, recalculate, self._model_names_list,
                                         apply_user_provided_initialization=not _in_recursion)

    def _run_composite_model(self, model, recalculate, model_names, apply_user_provided_initialization=False):
        with mot.configuration.config_context(RuntimeConfigurationAction(cl_environments=self._cl_envs,
                                                                         load_balancer=self._load_balancer)):
            with per_model_logging_context(os.path.join(self._output_folder, model.name)):
                self._logger.info('Using MDT version {}'.format(__version__))
                self._logger.info('Preparing for model {0}'.format(model.name))
                self._logger.info('Current cascade: {0}'.format(model_names))

                if apply_user_provided_initialization:
                    self._apply_user_provided_initialization_data(model)

                optimizer = self._optimizer or get_optimizer_for_model(model_names)

                if self._cl_device_indices is not None:
                    all_devices = get_cl_devices()
                    optimizer.cl_environments = [all_devices[ind] for ind in self._cl_device_indices]
                    optimizer.load_balancer = EvenDistribution()

                processing_strategy = get_processing_strategy('optimization', model_names=model_names,
                                                              tmp_dir=self._tmp_results_dir)

                fitter = SingleModelFit(model, self._problem_data, self._output_folder, optimizer, processing_strategy,
                                        recalculate=recalculate)
                results = fitter.run()

        map_results = get_all_image_data(os.path.join(self._output_folder, model.name))
        return results, map_results

    def _apply_user_provided_initialization_data(self, model):
        """Apply the initialization data to the model.

        This has the ability to initialize maps as well as fix maps.

        Args:
            model: the composite model we are preparing for fitting. Changes happen in place.
        """
        self._logger.info('Preparing model {0} with the user provided initialization data.'.format(model.name))
        self._initialization_data.apply_to_model(model, self._problem_data)


class SingleModelFit(object):

    def __init__(self, model, problem_data, output_folder, optimizer, processing_strategy, recalculate=False):
        """Fits a composite model.

         This does not accept cascade models. Please use the more general ModelFit class for all models,
         composite and cascade.

         Args:
             model (:class:`~mdt.models.composite.DMRICompositeModel`): An implementation of an composite model
                that contains the model we want to optimize.
             problem_data (:class:`~mdt.utils.DMRIProblemData`): The problem data object for the model
             output_folder (string): The path to the folder where to place the output.
                The resulting maps are placed in a subdirectory (named after the model name) in this output folder.
             optimizer (:class:`mot.cl_routines.optimizing.base.AbstractOptimizer`): The optimization routine to use.
             processing_strategy (:class:`~mdt.processing_strategies.ModelProcessingStrategy`): the processing strategy
                to use
             recalculate (boolean): If we want to recalculate the results if they are already present.
         """
        self.recalculate = recalculate

        self._model = model
        self._problem_data = problem_data
        self._output_folder = output_folder
        self._output_path = os.path.join(self._output_folder, self._model.name)
        self._optimizer = optimizer
        self._logger = logging.getLogger(__name__)
        self._processing_strategy = processing_strategy

        if not self._model.is_protocol_sufficient(problem_data.protocol):
            raise InsufficientProtocolError(
                'The given protocol is insufficient for this model. '
                'The reported errors where: {}'.format(self._model.get_protocol_problems(problem_data.protocol)))

    def run(self):
        """Fits the composite model and returns the results as ROI lists per map."""
        with per_model_logging_context(self._output_path):
            self._model.set_problem_data(self._problem_data)

            if self.recalculate:
                if os.path.exists(self._output_path):
                    list(map(os.remove, glob.glob(os.path.join(self._output_path, '*.nii*'))))
            else:
                if model_output_exists(self._model, self._output_folder):
                    maps = get_all_image_data(self._output_path)
                    self._logger.info('Not recalculating {} model'.format(self._model.name))
                    return create_roi(maps, self._problem_data.mask)

            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

            with self._logging():
                worker_generator = SimpleModelProcessingWorkerGenerator(
                    lambda *args: FittingProcessingWorker(self._optimizer, *args))

                results = self._processing_strategy.run(
                    self._model, self._problem_data, self._output_path, self.recalculate, worker_generator)

                self._write_protocol(self._model.get_problem_data().protocol)

        return results

    def _write_protocol(self, protocol):
        write_protocol(protocol, os.path.join(self._output_path, 'used_protocol.prtcl'))

    @contextmanager
    def _logging(self):
        """Adds logging information around the processing."""
        minimize_start_time = timeit.default_timer()
        self._logger.info('Fitting {} model'.format(self._model.name))

        yield

        run_time = timeit.default_timer() - minimize_start_time
        run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
        self._logger.info('Fitted {0} model with runtime {1} (h:m:s).'.format(self._model.name, run_time_str))
