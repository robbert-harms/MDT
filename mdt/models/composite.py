import inspect
import logging
from textwrap import dedent
import copy

import numpy as np

from mdt.configuration import get_active_post_processing
from mdt.deferred_mappings import DeferredFunctionDict
from mot.cl_routines.mapping.codec_runner import CodecRunner

from mdt.models.model_interfaces import MRIModelBuilder, MRIModelInterface
from mot.cl_routines.mapping.numerical_hessian import NumericalHessian
from mot.model_building.parameters import ProtocolParameter
from mot.statistics import deviance_information_criterions
from mot.utils import convert_data_to_dtype, KernelInputArray, get_class_that_defined_method, \
    hessian_to_covariance, covariance_to_correlations

from mdt.models.base import MissingProtocolInput, InsufficientShells
from mdt.models.base import DMRIOptimizable
from mdt.protocols import VirtualColumnB
from mdt.utils import create_roi, calculate_point_estimate_information_criterions, is_scalar, results_to_dict
from mot.cl_routines.mapping.calc_dependent_params import CalculateDependentParameters
from mot.cl_routines.mapping.loglikelihood_calculator import LogLikelihoodCalculator
from mot.cl_routines.mapping.waic_calculator import WAICCalculator
from mot.mcmc_diagnostics import multivariate_ess, univariate_ess
from mot.model_building.model_builders import SampleModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2014-10-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICompositeModel(SampleModelBuilder, DMRIOptimizable, MRIModelBuilder):

    def __init__(self, model_name, model_tree, likelihood_function, signal_noise_model=None, input_data=None,
                 enforce_weights_sum_to_one=True):
        """A model builder for a composite dMRI sample and optimization model.

        It implements some protocol check functions. These are used by the fit_model functions in MDT
        to check if the protocol is correct for the model we try to fit.

        Attributes:
            required_nmr_shells (int): Define the minimum number of unique shells necessary for this model.
                The default is false, which means that we don't check for this.
            _post_optimization_modifiers (list): the list with post optimization modifiers. Every element
                should contain a tuple with (str, Func) or (tuple, Func): where the first element is a single
                output name or a list with output names and the Func is a callback function that returns one or more
                output maps.
        """
        super(DMRICompositeModel, self).__init__(model_name, model_tree, likelihood_function, signal_noise_model,
                                                 input_data=input_data,
                                                 enforce_weights_sum_to_one=enforce_weights_sum_to_one)
        self._logger = logging.getLogger(__name__)
        self._original_input_data = None

        self._post_optimization_modifiers = []
        self._extra_optimization_maps_funcs = []

        if self._enforce_weights_sum_to_one:
            self._extra_optimization_maps_funcs.append(self._get_propagate_weights_uncertainty)

        self.nmr_parameters_for_bic_calculation = self.get_nmr_estimable_parameters()
        self.required_nmr_shells = False
        self._post_processing = get_active_post_processing()

    def build(self, problems_to_analyze=None):
        sample_model = super(DMRICompositeModel, self).build(problems_to_analyze)
        return BuildCompositeModel(sample_model,
                                   self._input_data.protocol,
                                   self._model_functions_info.get_estimable_parameters_list(),
                                   self.nmr_parameters_for_bic_calculation,
                                   self._post_optimization_modifiers,
                                   self._extra_optimization_maps_funcs,
                                   self._get_dependent_map_calculator(),
                                   self._get_fixed_parameter_maps(problems_to_analyze),
                                   self._get_proposal_state_names(),
                                   self._get_sampling_statistics(),
                                   self.get_free_param_names(),
                                   self.get_parameter_codec(),
                                   copy.deepcopy(self._post_processing))

    def update_active_post_processing(self, processing_type, settings):
        """Update the active post-processing semaphores.

        It is possible to control which post-processing routines get run by overwriting them using this method.
        For a list of post-processors, please see the default mdt configuration file under ``active_post_processing``.

        Args:
            processing_type (str): one of ``sampling`` or ``optimization``.
            settings (dict): the items to set in the post-processing information
        """
        self._post_processing[processing_type].update(settings)

    def get_active_post_processing(self):
        """Get a dictionary with the active post processing.

        This returns a dictionary with as first level the processing type and as second level the post-processing
        options.

        Returns:
            dict: the dictionary with the post-processing options for both sampling and optimization.
        """
        return copy.deepcopy(self._post_processing)

    def get_free_param_names(self):
        """Get the names of the free parameters"""
        return ['{}.{}'.format(m.name, p.name) for m, p in self._model_functions_info.get_estimable_parameters_list()]

    def set_input_data(self, input_data):
        """Overwrites the super implementation by adding a call to _prepare_input_data()."""
        self._check_data_consistency(input_data)
        self._original_input_data = input_data

        if input_data.gradient_deviations is not None:
            self._logger.info('Using the gradient deviations in the model optimization.')

        return super(DMRICompositeModel, self).set_input_data(self._prepare_input_data(input_data))

    def _get_kernel_data(self, problems_to_analyze):
        kernel_data = super(DMRICompositeModel, self)._get_kernel_data(problems_to_analyze)
        if self._input_data.gradient_deviations is not None:
            kernel_data['gradient_deviations'] = KernelInputArray(
                self._get_gradient_deviations(problems_to_analyze), ctype='mot_float_type')
        return kernel_data

    def _get_propagate_weights_uncertainty(self, results):
        std_names = ['{}.{}.std'.format(m.name, p.name) for (m, p) in self._model_functions_info.get_weights()]
        if len(std_names) > 1:
            if std_names[0] not in results and all(std in results for std in std_names[1:]):
                total = results[std_names[1]]**2
                for std in std_names[2:]:
                    total += results[std]**2
                return {std_names[0]: np.sqrt(total)}
        return {}

    def _get_gradient_deviations(self, problems_to_analyze):
        """Get the gradient deviation data for use in the kernel.

        This already adds the eye(3) matrix to every gradient deviation matrix, so we don't have to do it in the kernel.

        Please note that the gradient deviations matrix is in Fortran (column-major) order (per voxel).

        Returns:
            ndarray: the gradient deviations for the voxels being optimized (this function already takes care
                of the problems_to_analyze setting).
        """
        if len(self._input_data.gradient_deviations.shape) > 2:
            grad_dev = create_roi(self._input_data.gradient_deviations, self._input_data.mask)
        else:
            grad_dev = np.copy(self._input_data.gradient_deviations)

        grad_dev += np.eye(3).flatten()

        if problems_to_analyze is not None:
            grad_dev = grad_dev[problems_to_analyze, ...]

        return convert_data_to_dtype(grad_dev, 'mot_float_type*', self._get_mot_float_type())

    def get_required_protocol_names(self):
        """Get a list with the constant data names that are needed for this model to work.

        For example, an implementing diffusion MRI model might require the presence of the protocol parameter
        'g' and 'b'. This function should then return ('g', 'b').

        Returns:
            list: A list of columns names that need to be present in the protocol
        """
        return list(set([p.name for m, p in self._model_functions_info.get_model_parameter_list() if
                         isinstance(p, ProtocolParameter)]))

    def is_input_data_sufficient(self, input_data=None):
        return not self.get_input_data_problems(input_data=input_data)

    def get_input_data_problems(self, input_data=None):
        if input_data is None:
            input_data = self._input_data

        problems = []

        missing_columns = []
        for name in self.get_required_protocol_names():
            if name not in input_data.protocol and name not in input_data.static_maps:
                missing_columns.append(name)

        if missing_columns:
            problems.append(MissingProtocolInput(missing_columns))

        try:
            shells = input_data.protocol.get_nmr_shells()
            if shells < self.required_nmr_shells:
                problems.append(InsufficientShells(self.required_nmr_shells, shells))
        except KeyError:
            pass

        return problems

    def param_dict_to_array(self, volume_dict):
        params = [volume_dict['{}.{}'.format(m.name, p.name)] for m, p
                  in self._model_functions_info.get_estimable_parameters_list()]
        return np.concatenate([np.transpose(np.array([s]))
                               if len(s.shape) < 2 else s for s in params], axis=1)

    def _get_pre_model_expression_eval_code(self, problems_to_analyze):
        if self._can_use_gradient_deviations(problems_to_analyze):
            s = '''
                mot_float_type4 _new_gradient_vector_raw = _get_new_gradient_raw(g, data->gradient_deviations);
                mot_float_type _new_gradient_vector_length = length(_new_gradient_vector_raw);
                g = _new_gradient_vector_raw/_new_gradient_vector_length;
            '''
            s = dedent(s.replace('\t', ' '*4))

            if 'b' in list(self._get_protocol_data(problems_to_analyze).keys()):
                s += 'b *= _new_gradient_vector_length * _new_gradient_vector_length;' + "\n"

            if 'G' in list(self._get_protocol_data(problems_to_analyze).keys()):
                s += 'G *= _new_gradient_vector_length;' + "\n"

            return s

    def _get_pre_model_expression_eval_function(self, problems_to_analyze):
        if self._can_use_gradient_deviations(problems_to_analyze):
            return dedent('''
                #ifndef GET_NEW_GRADIENT_RAW
                #define GET_NEW_GRADIENT_RAW
                mot_float_type4 _get_new_gradient_raw(
                        mot_float_type4 g,
                        global const mot_float_type* const gradient_deviations){

                    const mot_float_type4 il_0 = (mot_float_type4)(gradient_deviations[0],
                                                                   gradient_deviations[3],
                                                                   gradient_deviations[6],
                                                                   0.0);

                    const mot_float_type4 il_1 = (mot_float_type4)(gradient_deviations[1],
                                                                   gradient_deviations[4],
                                                                   gradient_deviations[7],
                                                                   0.0);

                    const mot_float_type4 il_2 = (mot_float_type4)(gradient_deviations[2],
                                                                   gradient_deviations[5],
                                                                   gradient_deviations[8],
                                                                   0.0);

                    return (mot_float_type4)(dot(il_0, g), dot(il_1, g), dot(il_2, g), 0.0);
                }
                #endif //GET_NEW_GRADIENT_RAW
            '''.replace('\t', ' '*4))

    def _can_use_gradient_deviations(self, problems_to_analyze):
        return self._input_data.gradient_deviations is not None \
               and 'g' in list(self._get_protocol_data(problems_to_analyze).keys())

    def _prepare_input_data(self, input_data):
        """Update the input data to make it suitable for this model.

        Some of the models in diffusion MRI can only handle a subset of all volumes. For example, the S0 model
        can only work with the unweigthed signals, or the Tensor model that can only handle a b-value up to 1.5e9 s/m^2.

        Overwrite this function to limit the input data to a suitable range.

        Args:
            input_data (mdt.utils.MRIInputData): the input data set by the user

        Returns:
            mdt.utils.MRIInputData: either the same input data or a changed copy.
        """
        protocol = input_data.protocol
        indices = self._get_suitable_volume_indices(input_data)

        if len(indices) != protocol.length:
            self._logger.info('For this model, {}, we will use a subset of the protocol and DWI.'.format(self._name))
            self._logger.info('Using {} out of {} volumes, indices: {}'.format(
                len(indices), protocol.length, str(indices).replace('\n', '').replace('[  ', '[')))
            return input_data.get_subset(volumes_to_keep=indices)
        else:
            self._logger.info('No model protocol options to apply, using original protocol.')
        return input_data

    def _check_data_consistency(self, input_data):
        """Check the input data for any anomalies.

        We do this here so that implementing models can add additional consistency checks, or skip the checks.
        Also, by doing this here instead of in the Protocol class we ensure that the warnings end up in the log file.
        The final argument for putting this here is that I do not want any log output in the protocol tab.

        Args:
            input_data (mdt.utils.MRIInputData): the input data to analyze.
        """
        protocol = input_data.protocol

        def warn(warning):
            self._logger.warning('{}, proceeding with seemingly inconsistent values.'.format(warning))

        if 'TE' in protocol and 'TR' in protocol:
            if any(np.greater(protocol['TE'], protocol['TR'])):
                warn('Volumes detected where TE > TR')

        if 'TE' in protocol:
            if any(np.greater_equal(protocol['TE'], 1)):
                warn('Volumes detected where TE >= 1 second')

        if 'TR' in protocol:
            if any(np.greater_equal(protocol['TR'], 50)):
                warn('Volumes detected where TR >= 50 seconds')

        if 'delta' in protocol and 'Delta' in protocol and any(map(protocol.is_column_real, ['delta', 'Delta'])):
            if any(np.greater_equal(protocol['delta'], protocol['Delta'])):
                warn('Volumes detected where (small) delta >= (big) Delta')

        if 'Delta' in protocol and 'TE' in protocol:
            if any(np.greater_equal(protocol['Delta'], protocol['TE'])):
                warn('Volumes detected where (big) Delta >= TE')

        if all(map(protocol.is_column_real, ['G', 'delta', 'Delta', 'b'])):
            if not np.allclose(VirtualColumnB().get_values(protocol), protocol['b']):
                warn('Estimated b-values (from G, Delta, delta) differ from given b-values')

    def _get_suitable_volume_indices(self, input_data):
        """Usable in combination with _prepare_input_data, return the suitable volume indices.

        Get a list of volume indices that the model can use. This function is meant to remove common boilerplate code
        from writing your own _prepare_input_data object.

        Args:
            input_data (mot.model_building.input_data.InputData): the input data set by the user

        Returns:
            list: the list of indices we want to use for this model.
        """
        return list(range(input_data.protocol.length))

    def _get_dependent_map_calculator(self):
        """Get the calculation function to compute the maps for the dependent parameters."""
        estimable_parameters = self._model_functions_info.get_estimable_parameters_list(exclude_priors=True)
        dependent_parameters = self._model_functions_info.get_dependency_fixed_parameters_list(exclude_priors=True)

        if len(dependent_parameters):
            func = ''
            func += self._get_fixed_parameters_listing()
            func += self._get_estimable_parameters_listing()
            func += self._get_dependent_parameters_listing()

            dependent_parameter_names = [('{}.{}'.format(m.name, p.name).replace('.', '_'),
                                          '{}.{}'.format(m.name, p.name))
                                         for m, p in dependent_parameters]

            estimable_parameter_names = ['{}.{}'.format(m.name, p.name) for m, p in estimable_parameters]

            def calculator(model, results_dict):
                estimated_parameters = [results_dict[k] for k in estimable_parameter_names]
                cpd = CalculateDependentParameters(double_precision=self.double_precision)
                vals = cpd.calculate(model.get_kernel_data(), estimated_parameters, func, dependent_parameter_names)
                return results_to_dict(vals, [n[1] for n in dependent_parameter_names])
        else:
            def calculator(model, results_dict):
                return {}

        return calculator

    def _get_fixed_parameter_maps(self, problems_to_analyze):
        """In place add complete maps for the fixed parameters."""
        fixed_params = self._model_functions_info.get_value_fixed_parameters_list(exclude_priors=True)

        result = {}

        for (m, p) in fixed_params:
            name = '{}.{}'.format(m.name, p.name)
            value = self._model_functions_info.get_parameter_value(name)

            if is_scalar(value):
                result.update({name: np.tile(np.array([value]), (self._get_nmr_problems(problems_to_analyze),))})
            else:
                if problems_to_analyze is not None:
                    value = value[problems_to_analyze, ...]
                result.update({name: value})

        return result

    def _get_proposal_state_names(self):
        """Get a list of names for the adaptable proposal parameters.

        Returns:
            list: list of str with the name for each of the adaptable proposal parameters.
        """
        return_list = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            for param in p.sampling_proposal.get_parameters():
                if param.adaptable:
                    return_list.append('{}.{}.proposal.{}'.format(m.name, p.name, param.name))
        return return_list

    def _get_sampling_statistics(self):
        """Get a dictionary with for every parameter the sampling statistic function."""
        sampling_statistics = {}
        for parameter_name in self.get_free_param_names():
            parameter = self._model_functions_info.get_model_parameter_by_name(parameter_name)[1]
            sampling_statistics[parameter_name] = parameter.sampling_statistics
        return sampling_statistics


class BuildCompositeModel(MRIModelInterface):

    def __init__(self, wrapped_sample_model, protocol, estimable_parameters_list,
                 nmr_parameters_for_bic_calculation,
                 post_optimization_modifiers, extra_optimization_maps,
                 dependent_map_calculator, fixed_parameter_maps, proposal_state_names,
                 sampling_statistics, free_param_names, parameter_codec, post_processing):
        self._protocol = protocol
        self._estimable_parameters_list = estimable_parameters_list
        self.nmr_parameters_for_bic_calculation = nmr_parameters_for_bic_calculation
        self._post_optimization_modifiers = post_optimization_modifiers
        self._wrapped_sample_model = wrapped_sample_model
        self._dependent_map_calculator = dependent_map_calculator
        self._fixed_parameter_maps = fixed_parameter_maps
        self._proposal_state_names = proposal_state_names
        self._sampling_statistics = sampling_statistics
        self._free_param_names = free_param_names
        self._parameter_codec = parameter_codec
        self._post_processing = post_processing
        self._extra_optimization_maps = extra_optimization_maps

    def get_post_optimization_volume_maps(self, optimization_results):
        end_points = optimization_results.get_optimization_result()
        volume_maps = results_to_dict(end_points, self.get_free_param_names())
        volume_maps = self.post_process_optimization_maps(volume_maps, results_array=end_points)
        volume_maps.update({'ReturnCodes': optimization_results.get_return_codes()})
        return volume_maps

    def get_free_param_names(self):
        """Get the free parameter names of this build model.

        This is used by the processing strategies to create the results.
        """
        return self._free_param_names

    def get_proposal_state_names(self):
        """Get a list of names for the adaptable proposal parameters.

        Returns:
            list: list of str with the name for each of the adaptable proposal parameters.
        """
        return self._proposal_state_names

    def post_process_optimization_maps(self, results_dict, results_array=None, log_likelihoods=None):
        """This adds some extra optimization maps to the results dictionary.

        This function behaves as a procedure and as a function. The input dict can be updated in place, but it should
        also return a dict but that is merely for the purpose of chaining.

        This might change the results in the results dictionary with different parameter sets. For example,
        it is possible to reorient some maps or swap variables in the optimization maps.

        The current steps in this function:

            1) Add the maps for the dependent and fixed parameters
            2) Add the fixed maps to the results
            3) Apply each of the ``post_optimization_modifiers`` functions
            4) Add information criteria maps
            5) Calculate the covariance matrix according to the Fisher Information Matrix theory
            6) Add the additional results from the ``additional_result_funcs``

        Args:
            results_dict (dict): A dictionary with as keys the names of the parameters and as values the 1d maps with
                for each voxel the optimized parameter value. The given dictionary can be altered by this function.
            results_array (ndarray): if available, the results as an array instead of as a dictionary, if not given we
                will construct it in this function.
            log_likelihoods (ndarray): for every set of parameters the corresponding log likelihoods.
                If not provided they will be calculated from the parameters.

        Returns:
            dict: The same result dictionary but with updated values or with additional maps.
                It should at least return the results_dict.
        """
        if results_array is None:
            results_array = self._param_dict_to_array(results_dict)

        results_dict.update(self._dependent_map_calculator(self, results_dict))
        results_dict.update(self._fixed_parameter_maps)

        for routine in self._post_optimization_modifiers:
            results_dict.update(routine(results_dict))

        results_dict.update(self._get_post_optimization_information_criterion_maps(
            results_array, log_likelihoods=log_likelihoods))

        if self._post_processing['optimization']['covariance']:
            results_dict.update(self._calculate_hessian_covariance(results_array))

        for routine in self._extra_optimization_maps:
            try:
                results_dict.update(routine(results_dict))
            except KeyError as exc:
                logger = logging.getLogger(__name__)
                logger.error('Failed to execute extra optimization maps function, missing input: {}.'.format(str(exc)))

        return results_dict

    def get_post_sampling_maps(self, sampling_output):
        """Get the post sampling volume maps.

        This will return a dictionary mapping folder names to dictionaries with volumes to write.

        Args:
            sampling_output (mot.cl_routines.sampling.metropolis_hastings.MHSampleOutput): the output of the sampler

        Returns:
            dict: a dictionary with for every subdirectory the maps to save
        """
        samples = sampling_output.get_samples()

        mle_maps_cb, map_maps_cb = self._get_mle_map_statistics(sampling_output)

        items = {
            'maximum_likelihood': mle_maps_cb,
            'maximum_a_posteriori': map_maps_cb,
            'mh_state': lambda: self._get_mh_state_write_arrays(sampling_output.get_mh_state()),
            'proposal_state': lambda: results_to_dict(sampling_output.get_proposal_state(),
                                                      self.get_proposal_state_names()),
            'chain_end_point': lambda: results_to_dict(sampling_output.get_current_chain_position(),
                                                       self.get_free_param_names()),
        }
        if self._post_processing['sampling']['sample_statistics']:
            items.update({'sample_statistics': lambda: self._sample_statistics(sampling_output)})
        if self._post_processing['sampling']['univariate_ess']:
            items.update({'univariate_ess': lambda: self._get_univariate_ess(samples)})
        if self._post_processing['sampling']['multivariate_ess']:
            items.update({'multivariate_ess': lambda: self._get_multivariate_ess(samples)})

        return DeferredFunctionDict(items, cache=False)

    def _get_mh_state_write_arrays(self, mh_state):
        """Get the state of the Metropolis Hastings sampler to write out such that we can start later at that position.
        """
        sampling_counter = mh_state.get_proposal_state_sampling_counter()
        return {'proposal_state_sampling_counter': mh_state.get_proposal_state_sampling_counter(),
                'proposal_state_acceptance_counter': mh_state.get_proposal_state_acceptance_counter(),
                'online_parameter_variance': mh_state.get_online_parameter_variance(),
                'online_parameter_variance_update_m2': mh_state.get_online_parameter_variance_update_m2(),
                'online_parameter_mean': mh_state.get_online_parameter_mean(),
                'rng_state': mh_state.get_rng_state(),
                'nmr_samples_drawn': np.ones_like(sampling_counter) * mh_state.nmr_samples_drawn}

    def _sample_statistics(self, sampling_output):
        """Get some standard statistics about the samples.

        This returns the mean, standard deviation and covariances of the sampled parameters.

        Args:
            sampling_output (mot.cl_routines.sampling.metropolis_hastings.MHSampleOutput): the output of the sampler

        Returns:
            dict: the volume maps derived from the mean parameter value
        """
        samples = sampling_output.get_samples()

        volume_maps = self._get_univariate_parameter_statistics(samples)

        volume_maps.update(self._dependent_map_calculator(self, volume_maps))
        volume_maps.update(self._fixed_parameter_maps)
        volume_maps.update(self._get_post_optimization_information_criterion_maps(
            self._param_dict_to_array(volume_maps)))

        volume_maps.update(self._compute_covariances_correlations(samples, volume_maps))
        volume_maps.update(deviance_information_criterions(
            volume_maps['LogLikelihood'], sampling_output.get_log_likelihoods()))

        if self._post_processing['sampling']['waic']:
            volume_maps.update({'WAIC': np.nan_to_num(WAICCalculator().calculate(self, samples))})

        return volume_maps

    def _get_mle_map_statistics(self, sampling_output):
        """Get the maximum and corresponding volume maps of the MLE and MAP estimators.

        This computes the Maximum Likelihood Estimator and the Maximum A Posteriori in one run and computes from that
        the corresponding parameter and global post-optimization maps.

        Args:
            sampling_output (mot.cl_routines.sampling.metropolis_hastings.MHSampleOutput): the output of the sampler

        Returns:
            tuple(Func, Func): the function that generates the maps for the MLE and for the MAP estimators.
        """
        log_likelihoods = sampling_output.get_log_likelihoods()
        posteriors = log_likelihoods + sampling_output.get_log_priors()

        mle_indices = np.argmax(log_likelihoods, axis=1)
        map_indices = np.argmax(posteriors, axis=1)

        mle_values = log_likelihoods[range(log_likelihoods.shape[0]), mle_indices]
        map_values = posteriors[range(posteriors.shape[0]), map_indices]

        samples = sampling_output.get_samples()

        mle_samples = np.zeros(samples.shape[:2], dtype=samples.dtype)
        map_samples = np.zeros(samples.shape[:2], dtype=samples.dtype)

        for problem_ind in range(samples.shape[0]):
            mle_samples[problem_ind] = samples[problem_ind, :, mle_indices[problem_ind]]
            map_samples[problem_ind] = samples[problem_ind, :, map_indices[problem_ind]]

        def mle_maps():
            results = results_to_dict(mle_samples, self._free_param_names)
            maps = self.post_process_optimization_maps(results, results_array=mle_samples, log_likelihoods=mle_values)
            maps.update({'MaximumLikelihoodEstimator.indices': mle_indices})
            return maps

        def map_maps():
            results = results_to_dict(map_samples, self._free_param_names)
            maps = self.post_process_optimization_maps(results, results_array=map_samples, log_likelihoods=mle_values)
            maps.update({'MaximumAPosteriori': map_values,
                         'MaximumAPosteriori.indices': map_indices})
            return maps

        return mle_maps, map_maps

    def _get_univariate_ess(self, samples):
        """Get the univariate Effective Sample Size statistics for the given set of samples.

        Args:
            samples (ndarray): an (d, p, n) matrix for d problems, p parameters and n samples.

        Returns:
            dict: the volume maps with the univariate ESS statistics
        """
        uv_ess = univariate_ess(samples, method='standard_error')
        return results_to_dict(uv_ess, [a + '.UnivariateESS' for a in self.get_free_param_names()])

    def _get_multivariate_ess(self, samples):
        """Get the multivariate Effective Sample Size statistics for the given set of samples.

        Args:
            samples (ndarray): an (d, p, n) matrix for d problems, p parameters and n samples.

        Returns:
            dict: the volume maps with the ESS statistics
        """
        return {'MultivariateESS': np.nan_to_num(multivariate_ess(samples))}

    def _calculate_hessian_covariance(self, results_array):
        """Calculate the covariance and correlation matrix by taking the inverse of the Hessian.

        This first calculates/approximates the Hessian at each of the points using numerical differentiation.
        Afterwards we inverse the Hessian and compute a correlation matrix.

        Args:
            results_dict (dict): the list with the optimized points for each parameter
        """
        hessian = NumericalHessian().calculate(
            self,
            results_array,
            double_precision=True,
            step_ratio=2,
            nmr_steps=10,
            step_offset=2
        )
        covars = hessian_to_covariance(hessian)
        correlations = covariance_to_correlations(covars)

        param_names = ['{}.{}'.format(m.name, p.name) for m, p in self._estimable_parameters_list]

        results = {}
        for x_ind in range(len(param_names)):
            results[param_names[x_ind] + '.std'] = np.nan_to_num(np.sqrt(covars[:, x_ind, x_ind]))

            for y_ind in range(x_ind + 1, len(param_names)):
                results['Covariance_{}_to_{}'.format(param_names[x_ind], param_names[y_ind])] = covars[:, x_ind, y_ind]
                results['Correlation_{}_to_{}'.format(param_names[x_ind], param_names[y_ind])] = \
                    correlations[:, x_ind, y_ind]
        return results

    def _get_post_optimization_information_criterion_maps(self, results_array, log_likelihoods=None):
        """Add some final results maps to the results dictionary.

        This called by the function post_process_optimization_maps() as last call to add more maps.

        Args:
            results_array (ndarray): the results from model optimization.
            log_likelihoods (ndarray): for every set of parameters the corresponding log likelihoods.
                If not provided they will be calculated from the parameters.

        Returns:
            dict: the calculated information criterion maps
        """
        if log_likelihoods is None:
            log_likelihood_calc = LogLikelihoodCalculator()
            log_likelihoods = log_likelihood_calc.calculate(self, results_array)

        k = self.nmr_parameters_for_bic_calculation
        n = self.get_nmr_inst_per_problem()

        result = {'LogLikelihood': log_likelihoods}
        result.update(calculate_point_estimate_information_criterions(log_likelihoods, k, n))

        return result

    def _param_dict_to_array(self, volume_dict):
        params = [volume_dict['{}.{}'.format(m.name, p.name)] for m, p in self._estimable_parameters_list]
        return np.concatenate([np.transpose(np.array([s]))
                               if len(s.shape) < 2 else s for s in params], axis=1)

    def _get_univariate_parameter_statistics(self, samples):
        """Create the statistics for each of the parameters separately.

        Args:
            samples (ndarray): the sampled parameter maps, an (d, p, n) array with for d problems
                and p parameters n samples.

        Returns:
            dict: A dictionary with point estimates and statistical maps (like standard deviation) for each parameter.
        """
        expected_values = np.zeros((samples.shape[0], len(self.get_free_param_names())))
        all_stats = {}

        for ind, parameter_name in enumerate(self.get_free_param_names()):
            parameter_samples = samples[:, ind, ...]

            statistics = self._sampling_statistics[parameter_name].get_statistics(
                parameter_samples, self.get_lower_bounds()[ind], self.get_upper_bounds()[ind])

            expected_values[:, ind] = statistics.mean
            all_stats.update({'{}.std'.format(parameter_name): statistics.std})
            all_stats.update({'{}.{}'.format(parameter_name, statistic_key): v
                              for statistic_key, v in statistics.get_additional_statistics().items()})

        CodecRunner().encode_decode(expected_values, self.get_kernel_data(), self._parameter_codec,
                                    double_precision=self.double_precision)
        all_stats.update(results_to_dict(expected_values, self.get_free_param_names()))
        return all_stats

    def _compute_covariances_correlations(self, samples, marginal_statistics):
        """Compute the covariance and correlation between each of the free parameters.

        This uses the distance metric of the sample statistics to compute the covariance between two parameters.

        The covariance is computed using the sum of distances to the expected value for each of the parameters.
        The correlation is then computed as ``covar(a, b) / sqrt(std_a * std_b)`` for samples of
        two parameters ``a`` and ``b``.
        """
        results = {}

        for ind0 in range(len(self.get_free_param_names())):
            param_name0 = self.get_free_param_names()[ind0]
            distances0 = self._sampling_statistics[param_name0].get_distance_from_expected(
                samples[:, ind0, ...], marginal_statistics[param_name0])

            for ind1 in range(ind0 + 1, len(self.get_free_param_names())):
                param_name1 = self.get_free_param_names()[ind1]
                distances1 = self._sampling_statistics[param_name1].get_distance_from_expected(
                    samples[:, ind1, ...], marginal_statistics[param_name1])

                covar = np.sum(distances0 * distances1, axis=1) / (samples.shape[2] - 1)
                correlation = covar / (marginal_statistics[param_name0 + '.std']
                                       * marginal_statistics[param_name1 + '.std'])

                results['Covariance_{}_to_{}'.format(param_name0, param_name1)] = covar
                results['Correlation_{}_to_{}'.format(param_name0, param_name1)] = correlation
        return results

    def __getattribute__(self, item):
        try:
            value = super(BuildCompositeModel, self).__getattribute__(item)
            if hasattr(MRIModelInterface, item):
                if inspect.ismethod(value) or inspect.isfunction(value):
                    if not issubclass(get_class_that_defined_method(value), BuildCompositeModel):
                        raise NotImplementedError()
            return value
        except NotImplementedError:
            return getattr(super(BuildCompositeModel, self).__getattribute__('_wrapped_sample_model'), item)
        except AttributeError:
            return getattr(super(BuildCompositeModel, self).__getattribute__('_wrapped_sample_model'), item)
