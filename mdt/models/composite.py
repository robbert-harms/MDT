import inspect
import logging
from textwrap import dedent
import copy

import numpy as np

from mdt.configuration import get_active_post_processing
from mdt.deferred_mappings import DeferredFunctionDict
from mot.cl_routines.mapping.codec_runner import CodecRunner

from mdt.model_interfaces import MRIModelBuilder, MRIModelInterface
from mot.model_building.parameters import ProtocolParameter
from mot.utils import results_to_dict, convert_data_to_dtype, KernelInputArray, get_class_that_defined_method
from six import string_types

from mdt.models.base import MissingProtocolInput, InsufficientShells
from mdt.models.base import DMRIOptimizable
from mdt.protocols import VirtualColumnB
from mdt.utils import create_roi, calculate_point_estimate_information_criterions, is_scalar
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

        self.nmr_parameters_for_bic_calculation = self.get_nmr_estimable_parameters()
        self.required_nmr_shells = False
        self.active_post_processing = get_active_post_processing()

    def build(self, problems_to_analyze=None):
        sample_model = super(DMRICompositeModel, self).build(problems_to_analyze)
        return BuildCompositeModel(sample_model,
                                   self._input_data.protocol,
                                   self._model_functions_info.get_estimable_parameters_list(),
                                   self.nmr_parameters_for_bic_calculation,
                                   self._post_optimization_modifiers,
                                   self._get_dependent_map_calculator(),
                                   self._get_fixed_parameter_maps(problems_to_analyze),
                                   self._get_proposal_state_names(),
                                   self._get_sampling_statistics(),
                                   self.get_free_param_names(),
                                   self.get_parameter_codec(),
                                   copy.deepcopy(self.active_post_processing))

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

    def get_optimization_output_param_names(self):
        """Get a list with the names of the parameters, this is the list of keys to the titles and results.

        See get_free_param_names() for getting the names of the parameters that are actually being optimized.

        This should be a complete overview of all the maps returned from optimizing this model.

        Returns:
            list of str: a list with the parameter names
        """
        output_names = ['{}.{}'.format(m.name, p.name) for m, p in
                        self._model_functions_info.get_free_parameters_list()]

        for name, _ in self._post_optimization_modifiers:
            if isinstance(name, string_types):
                output_names.append(name)
            else:
                output_names.extend(name)

        return output_names

    def _get_kernel_data(self, problems_to_analyze):
        kernel_data = super(DMRICompositeModel, self)._get_kernel_data(problems_to_analyze)
        if self._input_data.gradient_deviations is not None:
            kernel_data['gradient_deviations'] = KernelInputArray(
                self._get_gradient_deviations(problems_to_analyze), ctype='mot_float_type')
        return kernel_data

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
                return cpd.calculate(model.get_kernel_data(), estimated_parameters, func, dependent_parameter_names)
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

    def __init__(self, wrapped_sample_model, protocol, estimable_parameters_list, nmr_parameters_for_bic_calculation,
                 post_optimization_modifiers, dependent_map_calculator, fixed_parameter_maps, proposal_state_names,
                 sampling_statistics, free_param_names, parameter_codec, active_post_processing):
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
        self._active_post_processing = active_post_processing

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
        r"""This adds some extra optimization maps to the results dictionary.

        This function behaves as a procedure and as a function. The input dict can be updated in place, but it should
        also return a dict but that is merely for the purpose of chaining.

        This might change the results in the results dictionary with different parameter sets. For example,
        it is possible to reorient some maps or swap variables in the optimization maps.

        The current steps in this function:

            1) Add the maps for the dependent and fixed parameters
            2) Add the extra maps defined in the models itself
            3) Apply each of the post_optimization_modifiers callback functions
            4) Add information criteria maps

        For more documentation see the base method.

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
        self._add_post_optimization_modifier_maps(results_dict)
        results_dict.update(self._get_post_optimization_information_criterion_maps(
            results_array, log_likelihoods=log_likelihoods))

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
            'sample_statistics': lambda: self._sample_statistics(sampling_output),
            'proposal_state': lambda: results_to_dict(sampling_output.get_proposal_state(),
                                                      self.get_proposal_state_names()),
            'chain_end_point': lambda: results_to_dict(sampling_output.get_current_chain_position(),
                                                       self.get_free_param_names()),
        }
        if self._active_post_processing['sampling']['univariate_ess']:
            items.update({'univariate_ess': lambda: self._get_univariate_ess(samples)})
        if self._active_post_processing['sampling']['multivariate_ess']:
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

        This typically returns the mean, standard deviation and covariances of each of the parameters.

        Args:
            sampling_output (mot.cl_routines.sampling.metropolis_hastings.MHSampleOutput): the output of the sampler

        Returns:
            dict: the volume maps derived from the mean parameter value
        """
        samples = sampling_output.get_samples()

        volume_maps = self._get_univariate_parameter_statistics(samples)
        volume_maps = self.post_process_optimization_maps(volume_maps)

        volume_maps.update(self._compute_covariances_correlations(samples, volume_maps))
        volume_maps.update(self._calculate_deviance_information_criterions(
            samples, volume_maps['LogLikelihood'], sampling_output.get_log_likelihoods()))

        if self._active_post_processing['sampling']['waic']:
            volume_maps.update({'WAIC': np.nan_to_num(WAICCalculator().calculate(self, samples))})

        return volume_maps

    def _get_mle_map_statistics(self, sampling_output):
        """Get the maximum and corresponding volume maps of the MLE and MAP estimators.

        This computes the Maximum Likelihood Estimator and the Maximum A Posteriori in one run and computes from that
        the corresponding parameter and general post-optimization maps.

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

    def _add_post_optimization_modifier_maps(self, results_dict):
        """Add the extra maps defined in the post optimization modifiers to the results."""
        for names, routine in self._post_optimization_modifiers:
            def callable():
                argspec = inspect.getfullargspec(routine)
                if len(argspec.args) > 1:
                    return routine(results_dict, self._protocol)
                else:
                    return routine(results_dict)

            if isinstance(names, string_types):
                results_dict[names] = callable()
            else:
                results_dict.update(zip(names, callable()))

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

    def _calculate_deviance_information_criterions(self, samples, mean_posterior_lls, ll_per_sample):
        r"""Calculates the Deviance Information Criteria (DIC) using two methods.

        This returns a dictionary returning the ``DIC_2002``, the ``DIC_2004`` and the ``DIC_Ando_2011`` method.
        The first is based on Spiegelhalter et al (2002), the second based on Gelman et al. (2004) and the last on
        Ando (2011). All cases differ in how they calculate model complexity, i.e. the effective number of parameters
        in the model. In all cases the model with the smallest DIC is preferred.

        All these DIC methods measure fitness using the deviance, which is, for a likelihood :math:`p(y | \theta)`
        defined as:

        .. math::

            D(\theta) = -2\log p(y|\theta)

        From this, the posterior mean deviance,

        .. math::

            \bar{D} = \mathbb{E}_{\theta}[D(\theta)]

        is then used as a measure of how well the model fits the data.

        The complexity, or measure of effective number of parameters, can be measured in see ways, see
        Spiegelhalter et al. (2002), Gelman et al (2004) and Ando (2011). The first method calculated the parameter
        deviance as:

        .. math::
            :nowrap:

            \begin{align}
            p_{D} &= \mathbb{E}_{\theta}[D(\theta)] - D(\mathbb{E}[\theta)]) \\
                  &= \bar{D} - D(\bar{\theta})
            \end{align}

        i.e. posterior mean deviance minus the deviance evaluated at the posterior mean of the parameters.

        The second method calculated :math:`p_{D}` as:

        .. math::

            p_{D} = p_{V} = \frac{1}{2}\hat{var}(D(\theta))

        i.e. half the variance of the deviance is used as an estimate of the number of free parameters in the model.

        The third method calculates the parameter deviance as:

        .. math::

            p_{D} = 2 \cdot (\bar{D} - D(\bar{\theta}))

        That is, twice the complexity of that of the first method.

        Finally, the DIC is (for all cases) defined as:

        .. math::

            DIC = \bar{D} + p_{D}

        Args:
            samples (ndarray): the samples, a (d, p, n) matrix with d problems, p parameters and n samples.
            mean_posterior_lls (ndarray): a 1d matrix containing the log likelihood for the average posterior
                point estimate.
            ll_per_sample (ndarray): a (d, n) array with for d problems the n log likelihoods.

        Returns:
            dict: a dictionary containing the ``DIC_2002``, the ``DIC_2004`` and the ``DIC_Ando_2011`` information
                criterion maps.
        """
        mean_deviance = -2 * np.mean(ll_per_sample, axis=1)
        deviance_at_mean = -2 * mean_posterior_lls

        pd_2002 = mean_deviance - deviance_at_mean
        pd_2004 = np.var(ll_per_sample, axis=1) / 2.0

        return {'DIC_2002': np.nan_to_num(mean_deviance + pd_2002),
                'DIC_2004': np.nan_to_num(mean_deviance + pd_2004),
                'DIC_Ando_2011': np.nan_to_num(mean_deviance + 2 * pd_2002)}

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

            expected_values[:, ind] = statistics.get_expected_value()
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
