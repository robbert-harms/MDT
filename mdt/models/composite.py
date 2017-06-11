import logging

import numpy as np
from mot.utils import results_to_dict
from six import string_types

from mdt.model_protocol_problem import MissingColumns, InsufficientShells
from mdt.models.base import DMRIOptimizable
from mdt.protocols import VirtualColumnB
from mdt.utils import create_roi, calculate_point_estimate_information_criterions, is_scalar
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
        """Create a composite dMRI sample model.

        This also implements the perturbation interface to allow perturbation of the data during meta-optimization.

        It furthermore implements some protocol check functions. These are used by the fit_model functions in MDT
        to check if the protocol is correct for the model we try to fit.

        Attributes:
            required_nmr_shells (int): Define the minimum number of unique shells necessary for this model.
                The default is false, which means that we don't check for this.
        """
        super(DMRICompositeModel, self).__init__(model_name, model_tree, evaluation_model, signal_noise_model,
                                                 problem_data=problem_data,
                                                 enforce_weights_sum_to_one=enforce_weights_sum_to_one)
        self.required_nmr_shells = False
        self._logger = logging.getLogger(__name__)
        self._original_problem_data = None
        self.nmr_parameters_for_bic_calculation = self.get_nmr_estimable_parameters()

    def set_problem_data(self, problem_data):
        """Overwrites the super implementation by adding a call to _prepare_problem_data() before the problem data is
        added to the model.
        """
        self._check_data_consistency(problem_data)
        self._original_problem_data = problem_data

        if problem_data.gradient_deviations is not None:
            self._logger.info('Using the gradient deviations in the model optimization.')

        return super(DMRICompositeModel, self).set_problem_data(self._prepare_problem_data(problem_data))

    def get_problem_data(self):
        """Get the problem data actually being used by this model.

        Returns:
            mdt.utils.DMRIProblemData: the problem data being used by this model
        """
        return self._problem_data

    def add_post_optimization_modifier(self, result_names, mod_routine):
        """Add a modification function that can update the results of model optimization.

        The mod routine should be a function accepting a dictionary as input and should return one or more maps of
        the same dimension as the maps in the dictionary. The idea is that the given ``mod_routine`` callback receives
        the result dictionary from the optimization routine and calculates new maps.
        These maps are then returned and the dictionary is updated with the returned maps as value and the here provided
        model_param_name as key.

        It is possible to add more than one modifier function. In that case, they are called in the order they
        were appended to this model.

        It is possible to add multiple maps in one modification routine, in that case the ``model_param_name`` should
        be a tuple of map names and the modification routine should also output a list of map names.

        Args:
            result_names (str or tuple of str): the name of the output(s) of the mod_routine.
                Example ``'Power2'`` or ``('Power2' 'Power3')``.
            mod_routine (python function): the callback function to apply on the results of the referenced parameter.
                Example: ``lambda d: d**2`` or ``lambda d: (d**2, d**3)``
        """
        self._post_optimization_modifiers.append((result_names, mod_routine))
        return self

    def add_post_optimization_modifiers(self, modifiers):
        """Add a list of modifier functions.

        The same as add_post_optimization_modifier() except that it accepts a list of modifiers.
        Every element in the list should be a tuple like (model_param_name, mod_routine)

        Args:
            modifiers (tuple or list): The list of modifiers to add (in the given order).
        """
        self._post_optimization_modifiers.extend(modifiers)
        return self

    def get_optimization_output_param_names(self):
        """See super class for details"""
        output_names = super(DMRICompositeModel, self).get_optimization_output_param_names()

        for name, _ in self._post_optimization_modifiers:
            if isinstance(name, string_types):
                output_names.append(name)
            else:
                output_names.extend(name)

        return output_names

    def add_extra_post_optimization_maps(self, results_dict):
        r"""This adds some extra optimization maps to the results dictionary.

        This function behaves as a procedure and as a function. The input dict can be updated in place, but it should
        also return a dict but that is merely for the purpose of chaining.

        Steps in finalizing the results dict:

            1) It first adds the maps for the dependent and fixed parameters
            2) Second it adds the extra maps defined in the models itself.
            3) Third it loops through the post_optimization_modifiers callback functions for the final updates.
            4) Finally it adds additional maps defined in this model subclass

        For more documentation see the base method.

        Args:
            results_dict (dict): A dictionary with as keys the names of the parameters and as values the 1d maps with
                for each voxel the optimized parameter value. The given dictionary can be altered by this function.

        Returns:
            dict: The same result dictionary but with updated values or with additional maps.
                It should at least return the results_dict.
        """
        self._add_dependent_parameter_maps(results_dict)
        self._add_fixed_parameter_maps(results_dict)
        self._add_post_optimization_modifier_maps(results_dict)
        self._add_post_optimization_information_criterion_maps(results_dict)
        return results_dict

    def get_post_sampling_maps(self, samples):
        """Get all the post sampling maps.

        Args:
            samples (ndarray): an (d, p, n) matrix for d problems, p parameters and n samples.

        Returns:
            dict: the volume maps with some basic post-sampling output
        """
        volume_maps = self.samples_to_statistics(samples)
        volume_maps = self.add_extra_post_optimization_maps(volume_maps)

        self._add_post_sampling_information_criterion_maps(samples, volume_maps)

        errors = ResidualCalculator().calculate(self, volume_maps)
        error_measures = ErrorMeasures(double_precision=self.double_precision).calculate(errors)
        volume_maps.update(error_measures)

        mv_ess = multivariate_ess(samples)
        volume_maps.update({'MultivariateESS': mv_ess})

        uv_ess = univariate_ess(samples, method='standard_error')
        uv_ess_maps = results_to_dict(uv_ess, [a + '.UnivariateESS' for a in self.get_free_param_names()])
        volume_maps.update(uv_ess_maps)

        return volume_maps

    def _get_variable_data(self):
        var_data_dict = super(DMRICompositeModel, self)._get_variable_data()
        if self._problem_data.gradient_deviations is not None:
            var_data_dict.update({'gradient_deviations': self._get_gradient_deviation_data_adapter()})
        return var_data_dict

    def _get_gradient_deviation_data_adapter(self):
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

        if self.problems_to_analyze is not None:
            grad_dev = grad_dev[self.problems_to_analyze, ...]

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

    def _get_pre_model_expression_eval_code(self):
        if self._can_use_gradient_deviations():
            s = '''
                mot_float_type4 _new_gradient_vector_raw = _get_new_gradient_raw(g, data->var_data_gradient_deviations);
                mot_float_type _new_gradient_vector_length = length(_new_gradient_vector_raw);
                g = _new_gradient_vector_raw/_new_gradient_vector_length;
            '''

            if 'b' in list(self._get_protocol_data().keys()):
                s += 'b *= _new_gradient_vector_length * _new_gradient_vector_length;' + "\n"

            if 'G' in list(self._get_protocol_data().keys()):
                s += 'G *= _new_gradient_vector_length;' + "\n"

            return s

    def _get_pre_model_expression_eval_function(self):
        if self._can_use_gradient_deviations():
            return '''
                #ifndef GET_NEW_GRADIENT_RAW
                #define GET_NEW_GRADIENT_RAW
                mot_float_type4 _get_new_gradient_raw(
                        mot_float_type4 g,
                        global const mot_float_type* const gradient_deviations){

                    const mot_float_type4 il_0 = (mot_float_type4)(gradient_deviations[0], gradient_deviations[3],
                                                             gradient_deviations[6], 0.0);

                    const mot_float_type4 il_1 = (mot_float_type4)(gradient_deviations[1], gradient_deviations[4],
                                                             gradient_deviations[7], 0.0);

                    const mot_float_type4 il_2 = (mot_float_type4)(gradient_deviations[2], gradient_deviations[5],
                                                             gradient_deviations[8], 0.0);

                    return (mot_float_type4)(dot(il_0, g), dot(il_1, g), dot(il_2, g), 0.0);
                }
                #endif //GET_NEW_GRADIENT_RAW
            '''

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

    def _add_dependent_parameter_maps(self, results_dict):
        """In place add complete maps for the dependent parameters."""
        estimable_parameters = self._model_functions_info.get_estimable_parameters_list(exclude_priors=True)
        dependent_parameters = self._model_functions_info.get_dependency_fixed_parameters_list(exclude_priors=True)

        if len(dependent_parameters):
            func = ''
            func += self._get_fixed_parameters_listing()
            func += self._get_estimable_parameters_listing()
            func += self._get_dependent_parameters_listing()

            estimable_params = ['{}.{}'.format(m.name, p.name) for m, p in estimable_parameters]
            estimated_parameters = [results_dict[k] for k in estimable_params]

            dependent_parameter_names = [('{}.{}'.format(m.name, p.name).replace('.', '_'),
                                          '{}.{}'.format(m.name, p.name))
                                         for m, p in dependent_parameters]

            cpd = CalculateDependentParameters(double_precision=self.double_precision)
            dependent_parameters = cpd.calculate(self, estimated_parameters, func, dependent_parameter_names)

            results_dict.update(dependent_parameters)

    def _add_fixed_parameter_maps(self, results_dict):
        """In place add complete maps for the fixed parameters."""
        fixed_params = self._model_functions_info.get_value_fixed_parameters_list(exclude_priors=True)

        for (m, p) in fixed_params:
            name = '{}.{}'.format(m.name, p.name)
            value = self._model_functions_info.get_parameter_value(name)

            if is_scalar(value):
                results_dict.update({name: np.tile(np.array([value]), (self.get_nmr_problems(),))})
            else:
                if self.problems_to_analyze is not None:
                    value = value[self.problems_to_analyze, ...]
                results_dict.update({name: value})

    def _add_post_optimization_modifier_maps(self, results_dict):
        """Add the extra maps defined in the post optimization modifiers to the results."""
        for names, routine in self._post_optimization_modifiers:
            if isinstance(names, string_types):
                results_dict[names] = routine(results_dict)
            else:
                results_dict.update(zip(names, routine(results_dict)))

    def _add_post_optimization_information_criterion_maps(self, results_dict):
        """Add some final results maps to the results dictionary.

        This called by the function add_extra_post_optimization_maps() as last call to add more maps.

        Args:
            results_dict (args): the results from model optmization. We are to modify this in-place.
        """
        log_likelihood_calc = LogLikelihoodCalculator()
        log_likelihoods = log_likelihood_calc.calculate(self, results_dict)

        k = self.nmr_parameters_for_bic_calculation
        n = self._problem_data.get_nmr_inst_per_problem()

        results_dict.update({'LogLikelihood': log_likelihoods})
        results_dict.update(calculate_point_estimate_information_criterions(log_likelihoods, k, n))

    def _add_post_sampling_information_criterion_maps(self, samples, results_dict):
        results_dict.update(self._calculate_deviance_information_criterions(samples, results_dict))
        results_dict.update({'WAIC': WAICCalculator().calculate(self, samples)})

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
        ll_per_sample = log_likelihood_calc.calculate(self, samples)

        mean_deviance = -2 * np.mean(ll_per_sample, axis=1)
        deviance_at_mean = -2 * results_dict['LogLikelihood']

        pd_2002 = mean_deviance - deviance_at_mean
        pd_2004 = np.var(ll_per_sample, axis=1) / 2.0

        return {'DIC_2002': mean_deviance + pd_2002,
                'DIC_2004': mean_deviance + pd_2004,
                'DIC_Ando_2011': mean_deviance + 2 * pd_2002}
