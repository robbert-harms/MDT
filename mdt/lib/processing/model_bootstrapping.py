import shutil
from contextlib import contextmanager
import logging
import os
import timeit
import time
import numpy as np
from numpy.lib.format import open_memmap
from mdt.configuration import gzip_sampling_results, get_processing_strategy
from mdt.lib.deferred_mappings import DeferredActionDict
from mdt.lib.nifti import write_all_as_nifti
from mdt.model_building.utils import ParameterDecodingWrapper
from mdt.utils import load_samples, per_model_logging_context, get_intermediate_results_path, create_roi, \
    restore_volumes, is_scalar, split_array_to_dict
from mdt.lib.input_data import ROIMRIInputData
from mdt.lib.processing.processing_strategies import SimpleModelProcessor
from mdt.lib.exceptions import InsufficientProtocolError
from mot import minimize
from mot.configuration import CLRuntimeInfo
from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Array, Zeros

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def compute_bootstrap(model, input_data, optimization_results, output_folder, bootstrap_method, optimization_method,
                      nmr_samples, tmp_dir, recalculate=False, keep_samples=True, optimizer_options=None,
                      bootstrap_options=None):
    """Sample a composite model using residual bootstrapping

    Args:
        model (:class:`~mdt.models.base.EstimableModel`): a composite model to sample
        input_data (:class:`~mdt.lib.input_data.MRIInputData`): The input data object with which the model
            is initialized before running
        optimization_results (dict or str): the optimization results, either a dictionary with results or the
            path to a folder.
        output_folder (string): The relative output path.
            The resulting maps are placed in a subdirectory (named after the model name) in this output folder.
        bootstrap_method (str): the bootstrap method to use, one of 'residual' or 'wild'.
        optimization_method (str): The optimization routine to use.
        nmr_samples (int): the number of samples we would like to return.
        tmp_dir (str): the preferred temporary storage dir
        recalculate (boolean): If we want to recalculate the results if they are already present.
        keep_samples (boolean): determines if we keep any of the chains. If set to False, the chains will
            be discarded after generating the mean and standard deviations.
        optimizer_options (dict): the additional optimization options
        bootstrap_options (dict): the bootstrap options
    """
    from mdt.__version__ import __version__
    logger = logging.getLogger(__name__)
    logger.info('Using MDT version {}'.format(__version__))
    logger.info('Preparing {} bootstrap for model {}'.format(bootstrap_method, model.name))

    output_folder = os.path.join(output_folder, model.name, '{}_bootstrap'.format(bootstrap_method))

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

    bootstrap_options = bootstrap_options or {}

    with per_model_logging_context(output_folder, overwrite=recalculate):
        with _log_info(logger, model.name):
            if bootstrap_method == 'residual':
                worker_class = ResidualBootstrappingProcessor
            else:
                worker_class = WildBootstrappingProcessor

            worker = worker_class(
                optimization_method,
                input_data,
                optimization_results,
                nmr_samples,
                model, input_data.mask, input_data.nifti_header, output_folder,
                get_intermediate_results_path(output_folder, tmp_dir), recalculate,
                keep_samples=keep_samples,
                optimizer_options=optimizer_options,
                **bootstrap_options
            )

            processing_strategy = get_processing_strategy('sampling')
            return processing_strategy.process(worker)


@contextmanager
def _log_info(logger, model_name):
    def calculate_run_days(runtime):
        if runtime > 24 * 60 * 60:
            return int(runtime // (24. * 60 * 60))
        return 0

    minimize_start_time = timeit.default_timer()
    logger.info('Bootstrapping {} model'.format(model_name))
    yield
    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = str(calculate_run_days(run_time)) + ':' + time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Residual bootstrapped {0} model with runtime {1} (d:h:m:s).'.format(model_name, run_time_str))


class BootstrappingProcessor(SimpleModelProcessor):

    class SampleChainNotStored:
        pass

    def __init__(self, optimization_method, input_data,
                 optimization_results, nmr_samples, model, mask, nifti_header,
                 output_dir, tmp_storage_dir, recalculate, keep_samples=True,
                 optimizer_options=None):
        """The processing worker for model sample.

        Args:
            optimization_method: the optimization routine to use
            optimization_results (dict): the starting point for the bootstrapping method
            nmr_samples (int): the number of samples we would like to return.
        """
        super().__init__(mask, nifti_header, output_dir, tmp_storage_dir, recalculate)
        self._logger = logging.getLogger(__name__)
        self._optimization_method = optimization_method
        self._input_data = input_data.get_subset(volumes_to_keep=model.get_used_volumes(input_data))
        self._roi_input_data = ROIMRIInputData.from_input_data(self._input_data)
        self._optimization_results = optimization_results
        self._nmr_samples = nmr_samples
        self._model = model
        self._write_volumes_gzipped = gzip_sampling_results()
        self._keep_samples = keep_samples
        self._logger = logging.getLogger(__name__)
        self._optimizer_options = optimizer_options
        self._sample_storage = None

        self._model.set_input_data(input_data)

        self._x_opt_array = self._model.param_dict_to_array(
            DeferredActionDict(lambda _, v: create_roi(v, self._input_data.mask), self._optimization_results))

        self._cl_runtime_info = CLRuntimeInfo()
        self._codec = self._model.get_mle_codec()
        self._lower_bounds, self._upper_bounds = self._codec.encode_bounds(self._model.get_lower_bounds(),
                                                                           self._model.get_upper_bounds())
        self._wrapper = ParameterDecodingWrapper(self._model.get_nmr_parameters(), self._codec.get_decode_function())
        self._objective_func = self._wrapper.wrap_objective_function(self._model.get_objective_function())
        self._constraints_func = self._wrapper.wrap_constraints_function(self._model.get_constraints_function())

    def _process(self, roi_indices, next_indices=None):
        """Apply the bootstrapping procedure on the given voxels.

        This method needs to be defined per bootstrapping scheme.
        """
        return NotImplementedError()

    def _single_optimization_run(self, y_star, x0, roi_indices):
        """Generate one new sample using a single run of the optimization routine."""
        self._roi_input_data.observations[roi_indices] = y_star
        self._model.set_input_data(self._roi_input_data, suppress_warnings=True)

        kernel_data_subset = self._model.get_kernel_data().get_subset(roi_indices)

        results = minimize(self._objective_func, x0,
                           method=self._optimization_method,
                           nmr_observations=self._model.get_nmr_observations(),
                           cl_runtime_info=self._cl_runtime_info,
                           data=self._wrapper.wrap_input_data(kernel_data_subset),
                           lower_bounds=self._get_bounds(self._lower_bounds, roi_indices),
                           upper_bounds=self._get_bounds(self._upper_bounds, roi_indices),
                           constraints_func=self._constraints_func,
                           options=self._optimizer_options)

        x_final_array = self._codec.decode(results['x'], kernel_data_subset)

        x_dict = split_array_to_dict(x_final_array, self._model.get_free_param_names())
        x_dict.update(self._model.get_post_optimization_output(x_final_array, roi_indices=roi_indices,
                                                               parameters_dict=x_dict))
        return x_dict

    def _get_signal_estimates(self, roi_indices):
        self._model.set_input_data(self._input_data, suppress_warnings=True)

        parameters = self._x_opt_array[roi_indices]
        kernel_data = {'data': self._model.get_kernel_data().get_subset(roi_indices),
                       'parameters': Array(parameters, ctype='mot_float_type'),
                       'estimates': Zeros((parameters.shape[0], self._model.get_nmr_observations()), 'mot_float_type')}

        eval_function_info = self._model.get_model_eval_function()
        simulate_function = SimpleCLFunction.from_string('''
                void simulate(void* data, local mot_float_type* parameters, global mot_float_type* estimates){
                    for(uint i = 0; i < ''' + str(self._model.get_nmr_observations()) + '''; i++){
                        estimates[i] = ''' + eval_function_info.get_cl_function_name() + '''(data, parameters, i);
                    }
                }
            ''', dependencies=[eval_function_info])

        simulate_function.evaluate(kernel_data, parameters.shape[0])
        return kernel_data['estimates'].get_data()

    def _get_bounds(self, bounds, roi_indices):
        return_values = []
        for el in bounds:
            if is_scalar(el):
                return_values.append(el)
            else:
                return_values.append(el[roi_indices])
        return tuple(return_values)

    def _store_sample(self, optimization_results, roi_indices, sample_ind):
        """Store the optimization results as a next sample."""
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        if self._sample_storage is None:
            self._sample_storage = {}
            for key, value in optimization_results.items():
                samples_path = os.path.join(self._output_dir, key + '.samples.npy')
                mode = 'w+'

                if os.path.isfile(samples_path):
                    mode = 'r+'
                    current_results = open_memmap(samples_path, mode='r')
                    if current_results.shape[1] != self._nmr_samples:
                        mode = 'w+'  # opening the memmap with w+ creates a new one
                    del current_results  # closes the memmap

                shape = [self._total_nmr_voxels, self._nmr_samples]
                if value.ndim > 1:
                    shape.extend(value.shape[1:])
                self._sample_storage[key] = open_memmap(samples_path, mode=mode, dtype=value.dtype, shape=tuple(shape))

        for key, value in optimization_results.items():
            self._sample_storage[key][roi_indices, sample_ind] = value

    def combine(self):
        super().combine()

        statistic_maps = {}
        for name in self._sample_storage:
            samples_path = os.path.join(self._output_dir, name + '.samples.npy')
            samples = open_memmap(samples_path, mode='r')
            statistic_maps[name] = np.mean(samples, axis=1)
            statistic_maps[name + '.std'] = np.std(samples, axis=1)

        write_all_as_nifti(restore_volumes(statistic_maps, self._mask),
                           os.path.join(self._output_dir, 'univariate_normal'),
                           nifti_header=self._nifti_header,
                           gzip=self._write_volumes_gzipped)

        write_all_as_nifti({'UsedMask': self._mask}, self._output_dir, nifti_header=self._nifti_header,
                           gzip=self._write_volumes_gzipped)

        if not self._keep_samples:
            for ind, name in enumerate(self._model.get_free_param_names()):
                os.remove(os.path.join(self._output_dir, name + '.samples.npy'))
        else:
            return load_samples(self._output_dir)


class ResidualBootstrappingProcessor(BootstrappingProcessor):

    def __init__(self, *args, **kwargs):
        """Compute bootstrap samples using residual bootstrapping. """
        super().__init__(*args, **kwargs)

    def _process(self, roi_indices, next_indices=None):
        y = self._get_signal_estimates(roi_indices)
        errors = self._input_data.observations[roi_indices] - y
        x0 = self._codec.encode(self._x_opt_array[roi_indices],
                                self._model.get_kernel_data().get_subset(roi_indices))

        for sample_ind in range(self._nmr_samples):
            if sample_ind % (self._nmr_samples * 0.1) == 0:
                self._logger.info('Processed samples {} from {}'.format(sample_ind, self._nmr_samples))

            errors_resampled = errors[range(errors.shape[1]),
                                      np.random.randint(0, errors.shape[1], size=errors.shape)]

            y_star = y + errors_resampled
            x_star = self._single_optimization_run(y_star, x0, roi_indices)
            self._store_sample(x_star, roi_indices, sample_ind)


class WildBootstrappingProcessor(BootstrappingProcessor):

    def __init__(self, *args, random_variable_method=None, **kwargs):
        """Compute bootstrap samples using the wild bootstrap

        The wild bootstrap has the additional option to define the random variable selection method. This
        is a callback function used to compute the random variable on which a new bootstrap is based.

        Args:
            random_variable_method (Callable[[Tuple], ndarray] or str), either a callable accepting a shape and
                returning an array of random variables of that shape, or a string with the name of a method to use.
                If a string is given, it can be one of:
                - normal for the standard normal distribution
                - mammen for the distribution proposed by Mammen 1993
                - simple for a distribution of v = -1 with p=0.5 and +1 for p=0.5
        """
        super().__init__(*args, **kwargs)
        self._random_variable_method = random_variable_method or (lambda v: np.random.randn(*v))

        if self._random_variable_method == 'normal':
            self._random_variable_method = (lambda v: np.random.randn(*v))
        elif self._random_variable_method == 'mannen':
            def distr(shape):
                cutoff = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
                r = np.random.rand(*shape)
                r[r < cutoff] = -(np.sqrt(5) - 1) / 2
                r[r >= cutoff] = (np.sqrt(5) + 1) / 2
                return r
            self._random_variable_method = distr
        elif self._random_variable_method == 'simple':
            def distr(shape):
                r = np.random.rand(*shape)
                r[r < 0.5] = -1
                r[r >= 0.5] = 1
                return r
            self._random_variable_method = distr

    def _process(self, roi_indices, next_indices=None):
        y = self._get_signal_estimates(roi_indices)
        errors = self._input_data.observations[roi_indices] - y
        x0 = self._codec.encode(self._x_opt_array[roi_indices],
                                self._model.get_kernel_data().get_subset(roi_indices))

        for sample_ind in range(self._nmr_samples):
            if sample_ind % (self._nmr_samples * 0.1) == 0:
                self._logger.info('Processed samples {} from {}'.format(sample_ind, self._nmr_samples))

            y_star = y + errors * self._random_variable_method(errors.shape)
            x_star = self._single_optimization_run(y_star, x0, roi_indices)
            self._store_sample(x_star, roi_indices, sample_ind)
