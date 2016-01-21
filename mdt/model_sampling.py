from contextlib import contextmanager

from mdt import __version__
import logging
import os
import shutil
import timeit
import time
import collections
from six import string_types
from mdt.IO import Nifti
from mdt.components_loader import get_model
from mdt.models.cascade import DMRICascadeModelInterface
from mdt.utils import create_roi, configure_per_model_logging, \
    ProtocolProblemError, model_output_exists, estimate_noise_std, get_cl_devices, get_model_config, \
    apply_model_protocol_options, get_processing_strategy, per_model_logging_context, SamplingProcessingWorker
from mot import runtime_configuration
from mot.cl_routines.sampling.metropolis_hastings import MetropolisHastings
from mot.load_balance_strategies import EvenDistribution
from mot.runtime_configuration import runtime_config_context

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelSampling(object):

    def __init__(self, model, problem_data, output_folder,
                 sampler=None, recalculate=False, cl_device_ind=None, double_precision=True,
                 model_protocol_options=None,
                 gradient_deviations=None, noise_std='auto',
                 initialize=True, initialize_using=None):
        """Sample a single model. This does not accept cascade models, only single models.

        Args:
            model: the model to sample
            problem_data (ProblemData): the problem data object which contains the dwi image, the dwi header, the
                brain_mask and the protocol to use.
            output_folder (string): The path to the folder where to place the output, we will make a subdir with the
                model name in it (for the optimization results) and then a subdir with the samples output.
            sampler (AbstractSampler): the sampler to use, if not set we will use MCMC
            recalculate (boolean): If we want to recalculate the results if they are already present.
            model_protocol_options (list of dict): specific model protocol options to use during fitting.
                This is for example used during batch fitting to limit the protocol for certain models.
                For instance, in the Tensor model we generally only want to use the lower b-values.
            cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
                utils.get_cl_devices().
            double_precision (boolean): if we would like to do the calculations in double precision
            noise_std (double or 'auto'): the noise level standard deviation. This is useful for model comparisons.
                    By default this is None and we set it to 1. If set to auto we try to estimate it using multiple
                    noise std calculators.
            gradient_deviations (str or ndarray): set of gradient deviations to use. In HCP WUMINN format.
            noise_std (double or 'auto'): the noise level standard deviation. This is useful for model comparisons.
                By default this is None and we set it to 1. If set to auto we try to estimate it using multiple
                noise std calculators.
            initialize (boolean): If we want to initialize the sampler with optimization output.
                This assumes that the optimization results are in the folder:
                    <output_folder>/<model_name>/
            initialize_using (None, str, or dict): If None, and initialize is True we will initialize from the
                optimization maps from a model with the same name. If a string is given and initialize is True we will
                interpret the string as a folder with the maps to load. If a dict is given and initialize is True we will
                initialize from the dict directly.

        Returns:
            the full chain of the optimization
        """
        if isinstance(model, string_types):
            model = get_model(model)

        if isinstance(model, DMRICascadeModelInterface):
            raise ValueError('The function \'sample_model()\' does not accept cascade models.')

        model.double_precision = double_precision

        self._model = model
        self._problem_data = problem_data
        self._output_folder = output_folder
        self._sampler = sampler
        self._recalculate = recalculate
        self._model_protocol_options = model_protocol_options
        self._logger = logging.getLogger(__name__)
        self._cl_device_indices = cl_device_ind
        self._noise_std = estimate_noise_std(noise_std, self._problem_data)
        self._initialize = initialize
        self._initialize_using = initialize_using

        if self._sampler is None:
            self._sampler = MetropolisHastings(runtime_configuration.runtime_config['cl_environments'],
                                               runtime_configuration.runtime_config['load_balancer'])

        if self._cl_device_indices is not None and not isinstance(self._cl_device_indices, collections.Iterable):
            self._cl_device_indices = [self._cl_device_indices]

        if gradient_deviations is not None:
            self._logger.info('Using given gradient deviations.')
            model.set_gradient_deviations(gradient_deviations)

        if not model.is_protocol_sufficient(self._problem_data.protocol):
            raise ProtocolProblemError(
                'The given protocol is insufficient for this model. '
                'The reported errors where: {}'.format(self._model.get_protocol_problems(self._problem_data.protocol)))

    def run(self):
        """Sample the given model, this does not return any results since those might be too large for memory."""
        cl_envs = None
        load_balancer = None
        if self._cl_device_indices is not None:
            all_devices = get_cl_devices()
            cl_envs = [all_devices[ind] for ind in self._cl_device_indices]
            load_balancer = EvenDistribution()

        with runtime_config_context(cl_environments=cl_envs, load_balancer=load_balancer):
            configure_per_model_logging(os.path.join(self._output_folder, self._model.name, 'samples'))

            self._logger.info('Using MDT version {}'.format(__version__))
            self._logger.info('Preparing for model {0}'.format(self._model.name))
            self._logger.info('Setting the noise standard deviation to {0}'.format(self._noise_std))
            self._model.evaluation_model.set_noise_level_std(self._noise_std)

            if self._cl_device_indices is not None:
                all_devices = get_cl_devices()
                self._sampler.cl_environments = [all_devices[ind] for ind in self._cl_device_indices]
                self._sampler.load_balancer = EvenDistribution()

            model_protocol_options = get_model_config([self._model.name], self._model_protocol_options)
            problem_data = apply_model_protocol_options(model_protocol_options, self._problem_data)

            processing_strategy = get_processing_strategy('sampling', self._model.name)

            sampler = SampleSingleModel(self._model, problem_data, self._output_folder, self._sampler,
                                        processing_strategy,
                                        recalculate=self._recalculate, initialize=self._initialize,
                                        initialize_using=self._initialize_using)

            sampler.run()


class SampleSingleModel(object):

    def __init__(self, model, problem_data, output_folder, sampler, processing_strategy,
                 recalculate=False, initialize=True, initialize_using=None):
        """Sample a single model.

        Please note that this function does not accept cascade models.

        This will place the output in the folder: <output_folder>/<model_name>/samples/

        Args:
            model (AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize.
            problem_data (DMRIProblemData): The problem data object with which the model is initialized before running
            output_folder (string): The full path to the folder where to place the output
            sampler (AbstractSampler): The sampling routine to use.
            processing_strategy (ModelProcessingStrategy): the processing strategy to use
            recalculate (boolean): If we want to recalculate the results if they are already present.
            initialize (boolean): If we want to initialize the sampler with optimization output.
                This assumes that the optimization results are in the folder:
                    <output_folder>/<model_name>/
            initialize_using (None, str, or dict): If None, and initialize is True we will initialize from the
                optimization maps from a model with the same name. If a string is given and initialize is True we will
                interpret the string as a folder with the maps to load. If a dict is given and initialize is True we will
                initialize from the dict directly.
        """
        self.recalculate = recalculate

        self._model = model
        self._problem_data = problem_data
        self._output_folder = output_folder
        self._output_path = os.path.join(output_folder, model.name, 'samples')
        self._sampler = sampler
        self._logger = logging.getLogger(__name__)
        self._processing_strategy = processing_strategy
        self._initialize = initialize
        self._initialize_using = initialize_using

        if not model.is_protocol_sufficient(problem_data.protocol):
            raise ProtocolProblemError(
                'The given protocol is insufficient for this model. '
                'The reported errors where: {}'.format(model.get_protocol_problems(problem_data.protocol)))

    def run(self):
        with per_model_logging_context(self._output_path):
            self._model.set_problem_data(self._problem_data)

            if self.recalculate:
                if os.path.exists(self._output_path):
                    shutil.rmtree(self._output_path)
            else:
                if model_output_exists(self._model, self._output_path + '/volume_maps/',
                                       append_model_name_to_path=False):
                    self._logger.info('Not recalculating {} model'.format(self._model.name))
                    return

            if not os.path.isdir(self._output_path):
                os.makedirs(self._output_path)

            with self._logging():
                self._model.set_initial_parameters(self._get_initialization_params())

                worker = SamplingProcessingWorker(self._sampler)

                self._processing_strategy.run(self._model, self._problem_data,
                                              self._output_path, self.recalculate, worker)

    def _get_initialization_params(self):
        logger = logging.getLogger(__name__)

        if self._initialize:
            maps = None
            if self._initialize_using is None:
                folder = os.path.join(self._output_folder, self._model.name)
                logger.info("Initializing sampler using maps in {}".format(folder))
                maps = Nifti.read_volume_maps(folder)
            elif isinstance(self._initialize_using, string_types):
                logger.info("Initializing sampler using maps in {}".format(self._initialize_using))
                maps = Nifti.read_volume_maps(self._initialize_using)
            elif isinstance(self._initialize_using, dict):
                logger.info("Initializing sampler using given maps.")
                maps = self._initialize_using

            if not maps:
                raise RuntimeError('No initialization maps found in the folder "{}"'.format(
                    os.path.join(self._output_folder, self._model.name)))

            init_params = create_roi(maps, self._problem_data.mask)
        else:
            init_params = {}

        return init_params

    @contextmanager
    def _logging(self):
        """Adds logging information around the processing."""
        minimize_start_time = timeit.default_timer()
        self._logger.info('Sampling {} model'.format(self._model.name))

        yield

        run_time = timeit.default_timer() - minimize_start_time
        run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
        self._logger.info('Sampled {0} model with runtime {1} (h:m:s).'.format(self._model.name, run_time_str))
