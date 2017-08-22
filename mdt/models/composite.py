import inspect
import logging
from textwrap import dedent

import numpy as np

from mot.model_interfaces import SampleModelInterface
from mot.utils import results_to_dict
from six import string_types

from mdt.model_protocol_problem import MissingColumns, InsufficientShells
from mdt.models.base import DMRIOptimizable
from mdt.protocols import VirtualColumnB
from mdt.utils import create_roi, calculate_point_estimate_information_criterions, is_scalar, spherical_to_cartesian
from mot.cl_data_type import SimpleCLDataType
from mot.cl_routines.mapping.calc_dependent_params import CalculateDependentParameters
from mot.cl_routines.mapping.error_measures import ErrorMeasures
from mot.cl_routines.mapping.loglikelihood_calculator import LogLikelihoodCalculator
from mot.cl_routines.mapping.residual_calculator import ResidualCalculator
from mot.cl_routines.mapping.waic_calculator import WAICCalculator
from mot.mcmc_diagnostics import multivariate_ess, univariate_ess
from mot.model_building.data_adapter import SimpleDataAdapter
from mot.model_building.model_builders import SampleModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2014-10-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICompositeModel(SampleModelBuilder, DMRIOptimizable):

    def __init__(self, model_name, model_tree, evaluation_model, signal_noise_model=None, problem_data=None,
                 enforce_weights_sum_to_one=True):
        """A model builder for a composite dMRI sample and optimization model.

        It implements some protocol check functions. These are used by the fit_model functions in MDT
        to check if the protocol is correct for the model we try to fit.

        Attributes:
            required_nmr_shells (int): Define the minimum number of unique shells necessary for this model.
                The default is false, which means that we don't check for this.
            _sampling_covar_excludes (list): the list of parameters (by model.param name) to exclude from the
                covariance calculation after sampling.
            _sampling_covar_extras (list): list with tuples containing information about additional variables
                to include in the sampling covariance matrix calculation. Every element of the given list should contain
                a tuple with three elements (list, list, Func): list of parameters required for the function, names of
                the resulting maps and the callback function itself.
            _post_optimization_modifiers (list): the list with post optimization modifiers. Every element
                should contain a tuple with (str, Func) or (tuple, Func): where the first element is a single
                output name or a list with output names and the Func is a callback function that returns one or more
                output maps.
        """
        super(DMRICompositeModel, self).__init__(model_name, model_tree, evaluation_model, signal_noise_model,
                                                 problem_data=problem_data,
                                                 enforce_weights_sum_to_one=enforce_weights_sum_to_one)
        self._logger = logging.getLogger(__name__)
        self._original_problem_data = None

        self._post_optimization_modifiers = []

        self.nmr_parameters_for_bic_calculation = self.get_nmr_estimable_parameters()
        self.required_nmr_shells = False
        self._sampling_covar_excludes = []
        self._sampling_covar_extras = []

    def build(self, problems_to_analyze=None):
        sample_model = super(DMRICompositeModel, self).build(problems_to_analyze)
        return BuildCompositeModel(sample_model,
                                   self._problem_data.protocol,
                                   self._model_functions_info.get_estimable_parameters_list(),
                                   self.nmr_parameters_for_bic_calculation,
                                   self._post_optimization_modifiers,
                                   self._get_dependent_map_calculator(),
                                   self._get_fixed_parameter_maps(problems_to_analyze),
                                   self._get_proposal_state_names(),
                                   self._get_sampling_statistics(),
                                   self._sampling_covar_excludes,
                                   self._sampling_covar_extras)

    def set_problem_data(self, problem_data):
        """Overwrites the super implementation by adding a call to _prepare_problem_data()."""
        self._check_data_consistency(problem_data)
        self._original_problem_data = problem_data

        if problem_data.gradient_deviations is not None:
            self._logger.info('Using the gradient deviations in the model optimization.')

        return super(DMRICompositeModel, self).set_problem_data(self._prepare_problem_data(problem_data))

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

    def _get_variable_data(self, problems_to_analyze):
        var_data_dict = super(DMRICompositeModel, self)._get_variable_data(problems_to_analyze)
        if self._problem_data.gradient_deviations is not None:
            var_data_dict['gradient_deviations'] = self._get_gradient_deviation_data_adapter(problems_to_analyze)
        return var_data_dict

    def _get_gradient_deviation_data_adapter(self, problems_to_analyze):
        """Get the gradient deviation data for use in the kernel.

        This already adds the eye(3) matrix to every gradient deviation matrix, so we don't have to do it in the kernel.

        Please note that the gradient deviations matrix is in Fortran (column-major) order (per voxel).

        Returns:
            ndarray: the gradient deviations for the voxels being optimized (this function already takes care
                of the problems_to_analyze setting).
        """
        if len(self._problem_data.gradient_deviations.shape) > 2:
            grad_dev = create_roi(self._problem_data.gradient_deviations, self._problem_data.mask)
        else:
            grad_dev = np.copy(self._problem_data.gradient_deviations)

        grad_dev += np.eye(3).flatten()

        if problems_to_analyze is not None:
            grad_dev = grad_dev[problems_to_analyze, ...]

        return SimpleDataAdapter(grad_dev, SimpleCLDataType.from_string('mot_float_type*'), self._get_mot_float_type(),
                                 allow_local_pointer=False)

    def is_protocol_sufficient(self, protocol=None):
        """See ProtocolCheckInterface"""
        return not self.get_protocol_problems(protocol=protocol)

    def get_protocol_problems(self, protocol=None):
        """See ProtocolCheckInterface"""
        if protocol is None:
            protocol = self._problem_data.protocol

        problems = []

        missing_columns = [name for name in self.get_required_protocol_names() if not protocol.has_column(name)]
        if missing_columns:
            problems.append(MissingColumns(missing_columns))

        try:
            shells = protocol.get_nmr_shells()
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

    def _get_pre_model_expression_eval_code(self):
        if self._can_use_gradient_deviations():
            s = '''
                mot_float_type4 _new_gradient_vector_raw = _get_new_gradient_raw(g, data->var_data_gradient_deviations);
                mot_float_type _new_gradient_vector_length = length(_new_gradient_vector_raw);
                g = _new_gradient_vector_raw/_new_gradient_vector_length;
            '''
            s = dedent(s.replace('\t', ' '*4))

            if 'b' in list(self._get_protocol_data().keys()):
                s += 'b *= _new_gradient_vector_length * _new_gradient_vector_length;' + "\n"

            if 'G' in list(self._get_protocol_data().keys()):
                s += 'G *= _new_gradient_vector_length;' + "\n"

            return s

    def _get_pre_model_expression_eval_function(self):
        if self._can_use_gradient_deviations():
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

    def _can_use_gradient_deviations(self):
        return self._problem_data.gradient_deviations is not None \
               and 'g' in list(self._get_protocol_data().keys())

    def _prepare_problem_data(self, problem_data):
        """Update the problem data to make it suitable for this model.

        Some of the models in diffusion MRI can only handle a subset of all volumes. For example, the S0 model
        can only work with the unweigthed signals, or the Tensor model that can only handle a b-value up to 1.5e9 s/m^2.

        Overwrite this function to limit the problem data to a suitable range.

        Args:
            problem_data (mdt.utils.DMRIProblemData): the problem data set by the user

        Returns:
            mdt.utils.DMRIProblemData: either the same problem data or a changed copy.
        """
        protocol = problem_data.protocol
        indices = self._get_suitable_volume_indices(problem_data)

        if len(indices) != protocol.length:
            self._logger.info('For this model, {}, we will use a subset of the protocol and DWI.'.format(self._name))
            self._logger.info('Using {} out of {} volumes, indices: {}'.format(
                len(indices), protocol.length, str(indices).replace('\n', '').replace('[  ', '[')))
            return problem_data.get_subset(volumes_to_keep=indices)
        else:
            self._logger.info('No model protocol options to apply, using original protocol.')
        return problem_data

    def _check_data_consistency(self, problem_data):
        """Check the problem data for any strange anomalies.

        We do this here so that implementing models can add additional consistency checks, or skip the checks.
        Also, by doing this here instead of in the Protocol class we ensure that the warnings end up in the log file.
        The final argument for putting this here is that I do not want any log output in the protocol tab.

        Args:
            problem_data (DMRIProblemData): the problem data to analyze.
        """
        protocol = problem_data.protocol

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

    def _get_suitable_volume_indices(self, problem_data):
        """Usable in combination with _prepare_problem_data, return the suitable volume indices.

        Get a list of volume indices that the model can use. This function is meant to remove common boilerplate code
        from writing your own _prepare_problem_data object.

        Args:
            problem_data (DMRIProblemData): the problem data set by the user

        Returns:
            list: the list of indices we want to use for this model.
        """
        return list(range(problem_data.protocol.length))

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
                return cpd.calculate(model, estimated_parameters, func, dependent_parameter_names)
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


class BuildCompositeModel(SampleModelInterface):

    def __init__(self, wrapped_sample_model, protocol,
                 estimable_parameters_list, nmr_parameters_for_bic_calculation,
                 post_optimization_modifiers, dependent_map_calculator, fixed_parameter_maps, proposal_state_names,
                 sampling_statistics, sampling_covar_excludes, sampling_covar_extras):
        self._protocol = protocol
        self._estimable_parameters_list = estimable_parameters_list
        self.nmr_parameters_for_bic_calculation = nmr_parameters_for_bic_calculation
        self._post_optimization_modifiers = post_optimization_modifiers
        self._wrapped_sample_model = wrapped_sample_model
        self._dependent_map_calculator = dependent_map_calculator
        self._fixed_parameter_maps = fixed_parameter_maps
        self._proposal_state_names = proposal_state_names
        self._sampling_statistics = sampling_statistics
        self._sampling_covar_excludes = sampling_covar_excludes
        self._sampling_covar_extras = sampling_covar_extras

    def get_proposal_state_names(self):
        """Get a list of names for the adaptable proposal parameters.

        Returns:
            list: list of str with the name for each of the adaptable proposal parameters.
        """
        return self._proposal_state_names

    def post_process_optimization_maps(self, results_dict, results_array=None):
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
            problems_to_analyze (ndarray): the subset of voxels we are analyzing
            results_array (ndarray): if available, the results as an array instead of as a dictionary, if not given we
                will construct it in this function.

        Returns:
            dict: The same result dictionary but with updated values or with additional maps.
                It should at least return the results_dict.
        """
        if results_array is None:
            results_array = self._param_dict_to_array(results_dict)

        results_dict.update(self._dependent_map_calculator(self, results_dict))
        results_dict.update(self._fixed_parameter_maps)
        self._add_post_optimization_modifier_maps(results_dict)
        results_dict.update(self._get_post_optimization_information_criterion_maps(results_array))
        return results_dict

    def get_post_sampling_maps(self, samples):
        """Get all the post sampling maps.

        Args:
            samples (ndarray): an (d, p, n) matrix for d problems, p parameters and n samples.

        Returns:
            dict: the volume maps with some basic post-sampling output
        """
        volume_maps = self._get_univariate_parameter_statistics(samples)

        results_array = self._param_dict_to_array(volume_maps)

        volume_maps = self.post_process_optimization_maps(volume_maps, results_array=results_array)

        self._add_post_sampling_information_criterion_maps(samples, volume_maps)

        errors = ResidualCalculator().calculate(self, results_array)
        errors = np.nan_to_num(errors)
        error_measures = ErrorMeasures(double_precision=self.double_precision).calculate(errors)
        volume_maps.update(error_measures)

        mv_ess = np.nan_to_num(multivariate_ess(samples))
        volume_maps.update({'MultivariateESS': mv_ess})

        uv_ess = univariate_ess(samples, method='standard_error')
        uv_ess_maps = results_to_dict(uv_ess, [a + '.UnivariateESS' for a in self.get_free_param_names()])
        volume_maps.update(uv_ess_maps)

        return volume_maps

    def get_multivariate_sampling_statistic(self, samples):
        """Get the multivariate statistics for the given samples.

        Calculates a single multivariate statistic based on the samples. This function might remove some of the
        parameters from the samples and might add some additional parameters at its own discretion.

        Args:
            samples (ndarray): the matrix with the samples

        Returns:
            dict: the elements forming the multivariate statistic.
        """
        params_to_exclude = set(self.get_free_param_names()).intersection(self._sampling_covar_excludes)
        param_names = [name for name in self.get_free_param_names() if name not in params_to_exclude]

        lti = np.array(np.tril_indices(len(param_names))).transpose()
        result_names = ['Covariance_{}_to_{}'.format(param_names[row], param_names[column]) for row, column in lti]
        result_matrices = {result_name: np.zeros((samples.shape[0], 1)) for result_name in result_names}

        for input_params, output_params, func in self._sampling_covar_extras:
            input_indices = []
            for p in input_params:
                if p in self.get_free_param_names():
                    input_indices.append(self.get_free_param_names().index(p))

            if input_indices:
                param_names.extend(output_params)
                inputs = [samples[:, ind, :] for ind in input_indices]
                samples = np.column_stack([samples, func(*inputs)])

        indices_to_remove = tuple(set(self.get_free_param_names().index(p) for p in params_to_exclude))
        samples = np.delete(samples, indices_to_remove, axis=1)

        for voxel_ind in range(samples.shape[0]):
            covar_matrix = np.cov(samples[voxel_ind, :])

            for names_ind, (row, column) in enumerate(lti):
                result_matrices[result_names[names_ind]][voxel_ind] = covar_matrix[row, column]

        for param_ind, param_name in enumerate(param_names):
            result_matrices['Mean_' + param_name] = np.mean(samples[:, param_ind, :], axis=1)

        return result_matrices

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

    def _get_post_optimization_information_criterion_maps(self, results_array):
        """Add some final results maps to the results dictionary.

        This called by the function post_process_optimization_maps() as last call to add more maps.

        Args:
            results_array (ndarray): the results from model optimization.

        Returns:
            dict: the calculated information criterion maps
        """
        log_likelihood_calc = LogLikelihoodCalculator()
        log_likelihoods = log_likelihood_calc.calculate(self, results_array)

        k = self.nmr_parameters_for_bic_calculation
        n = self.get_nmr_inst_per_problem()

        result = {'LogLikelihood': log_likelihoods}
        result.update(calculate_point_estimate_information_criterions(log_likelihoods, k, n))

        return result

    def _add_post_sampling_information_criterion_maps(self, samples, results_dict):
        results_dict.update(self._calculate_deviance_information_criterions(samples, results_dict))
        results_dict.update({'WAIC': np.nan_to_num(WAICCalculator().calculate(self, samples))})

    def _calculate_deviance_information_criterions(self, samples, results_dict):
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
            results_dict (dict): the dictionary with the point estimates, should contain the ``LogLikelihood`` map
                with a point estimate LL.

        Returns:
            dict: a dictionary containing the ``DIC_2002``, the ``DIC_2004`` and the ``DIC_Ando_2011`` information
                criterion maps.
        """
        log_likelihood_calc = LogLikelihoodCalculator()
        ll_per_sample = np.nan_to_num(log_likelihood_calc.calculate(self, samples))

        mean_deviance = -2 * np.mean(ll_per_sample, axis=1)
        deviance_at_mean = -2 * results_dict['LogLikelihood']

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
        results = {}

        for ind, parameter_name in enumerate(self.get_free_param_names()):
            parameter_samples = samples[:, ind, ...]

            statistics = self._sampling_statistics[parameter_name].get_statistics(parameter_samples)

            results[parameter_name] = statistics.get_point_estimate()
            results.update({'{}.{}'.format(parameter_name, statistic_key): v
                            for statistic_key, v in statistics.get_additional_statistics().items()})
        return results

    @property
    def name(self):
        return self._wrapped_sample_model.name

    @property
    def double_precision(self):
        return self._wrapped_sample_model.double_precision

    def get_free_param_names(self):
        return self._wrapped_sample_model.get_free_param_names()

    def get_kernel_data_info(self):
        return self._wrapped_sample_model.get_kernel_data_info()

    def get_nmr_problems(self):
        return self._wrapped_sample_model.get_nmr_problems()

    def get_nmr_inst_per_problem(self):
        return self._wrapped_sample_model.get_nmr_inst_per_problem()

    def get_nmr_estimable_parameters(self):
        return self._wrapped_sample_model.get_nmr_estimable_parameters()

    def get_pre_eval_parameter_modifier(self):
        return self._wrapped_sample_model.get_pre_eval_parameter_modifier()

    def get_model_eval_function(self):
        return self._wrapped_sample_model.get_model_eval_function()

    def get_residual_per_observation_function(self):
        return self._wrapped_sample_model.get_residual_per_observation_function()

    def get_objective_per_observation_function(self):
        return self._wrapped_sample_model.get_objective_per_observation_function()

    def get_initial_parameters(self):
        return self._wrapped_sample_model.get_initial_parameters()

    def get_lower_bounds(self):
        return self._wrapped_sample_model.get_lower_bounds()

    def get_upper_bounds(self):
        return self._wrapped_sample_model.get_upper_bounds()

    def get_proposal_state(self):
        return self._wrapped_sample_model.get_proposal_state()

    def get_log_likelihood_per_observation_function(self, full_likelihood=True):
        return self._wrapped_sample_model.get_log_likelihood_per_observation_function(full_likelihood=full_likelihood)

    def is_proposal_symmetric(self):
        return self._wrapped_sample_model.is_proposal_symmetric()

    def get_proposal_logpdf(self, address_space_proposal_state='private'):
        return self._wrapped_sample_model.get_proposal_logpdf(address_space_proposal_state)

    def get_proposal_function(self, address_space_proposal_state='private'):
        return self._wrapped_sample_model.get_proposal_function(address_space_proposal_state)

    def get_proposal_state_update_function(self, address_space='private'):
        return self._wrapped_sample_model.get_proposal_state_update_function(address_space)

    def proposal_state_update_uses_variance(self):
        return self._wrapped_sample_model.proposal_state_update_uses_variance()

    def get_log_prior_function(self, address_space_parameter_vector='private'):
        return self._wrapped_sample_model.get_log_prior_function(address_space_parameter_vector)

    def get_metropolis_hastings_state(self):
        return self._wrapped_sample_model.get_metropolis_hastings_state()
