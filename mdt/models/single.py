import logging
from copy import deepcopy

import numpy as np

from mdt import utils
from mdt.components_loader import ComponentConfig, ComponentBuilder, method_binding_meta
from mdt.model_protocol_problem import MissingColumns, InsufficientShells
from mdt.models.base import DMRIOptimizable
from mdt.models.parsers.SingleModelExpressionParser import parse
from mdt.protocols import VirtualColumnB
from mdt.utils import create_roi
from mot.cl_data_type import CLDataType
from mot.model_building.cl_functions.model_functions import Weight
from mot.cl_routines.mapping.loglikelihood_calculator import LogLikelihoodCalculator
from mot.data_adapters import SimpleDataAdapter
from mot.model_building.evaluation_models import OffsetGaussianEvaluationModel
from mot.model_building.model_builders import SampleModelBuilder
from mot.model_building.parameter_functions.dependencies import WeightSumToOneRule
from mot.model_building.trees import CompartmentModelTree

__author__ = 'Robbert Harms'
__date__ = "2014-10-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRISingleModel(SampleModelBuilder, DMRIOptimizable):

    def __init__(self, model_name, model_tree, evaluation_model, signal_noise_model=None, problem_data=None):
        """Create a composite dMRI sample model.

        This also implements the perturbation interface to allow perturbation of the data during meta-optimization.

        It furthermore implements some protocol check functions. These are used by the fit_model functions in MDT
        to check if the protocol is correct for the model we try to fit.

        Attributes:
            required_nmr_shells (int): Define the minimum number of unique shells necessary for this model.
                The default is false, which means that we don't check for this.
        """
        super(DMRISingleModel, self).__init__(model_name, model_tree, evaluation_model, signal_noise_model,
                                              problem_data=problem_data)
        self.required_nmr_shells = False
        self._logger = logging.getLogger(__name__)
        self._original_problem_data = None

    def set_problem_data(self, problem_data):
        """Overwrites the super implementation by adding a call to _prepare_problem_data() before the problem data is
        added to the model.
        """
        self._check_data_consistency(problem_data)
        self._original_problem_data = problem_data
        return super(DMRISingleModel, self).set_problem_data(self._prepare_problem_data(problem_data))

    def get_problems_var_data(self):
        var_data_dict = super(DMRISingleModel, self).get_problems_var_data()

        if self._problem_data.gradient_deviations is not None:
            if len(self._problem_data.gradient_deviations.shape) > 2:
                grad_dev = create_roi(self._problem_data.gradient_deviations, self._problem_data.mask)
            else:
                grad_dev = np.copy(self._problem_data.gradient_deviations)

            self._logger.info('Using the gradient deviations in the model optimization.')

            # adds the eye(3) matrix to every grad dev, so we don't have to do it in the kernel.
            # Flattening an eye(3) matrix gives the same result with F and C ordering, I nevertheless put it here
            # to emphasize that the gradient deviations matrix is in Fortran (column-major) order.
            grad_dev += np.eye(3).flatten(order='F')

            if self.problems_to_analyze is not None:
                grad_dev = grad_dev[self.problems_to_analyze, ...]

            adapter = SimpleDataAdapter(grad_dev, CLDataType.from_string('mot_float_type*'), self._get_mot_float_type())
            var_data_dict.update({'gradient_deviations': adapter})

        return var_data_dict

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

    def get_abstract_model_function(self):
        """Get the abstract diffusion model function computed by this model.

        Returns:
            str: the abstract model function of this class
        """
        return self._model_tree

    def _set_default_dependencies(self):
        super(DMRISingleModel, self)._set_default_dependencies()
        names = [w.name + '.w' for w in self._get_weight_models()]
        if len(names):
            self.add_parameter_dependency(names[0], WeightSumToOneRule(names[1:]))

    def _get_weight_models(self):
        return [n.data for n in self._model_tree.leaves if isinstance(n.data, Weight)]

    def _get_pre_model_expression_eval_code(self):
        if self._can_use_gradient_deviations():
            s = '''
                mot_float_type4 _new_gradient_vector_raw = _get_new_gradient_raw(g, data->var_data_gradient_deviations);
                mot_float_type _new_gradient_vector_length = length(_new_gradient_vector_raw);
                g = _new_gradient_vector_raw/_new_gradient_vector_length;
            '''
            if 'b' in list(self.get_problems_protocol_data().keys()):
                s += 'b *= _new_gradient_vector_length * _new_gradient_vector_length;' + "\n"

            if 'G' in list(self.get_problems_protocol_data().keys()):
                s += 'G *= _new_gradient_vector_length;' + "\n"

            if 'q' in list(self.get_problems_protocol_data().keys()):
                s += 'q *= _new_gradient_vector_length;' + "\n"

            return s

    def _get_pre_model_expression_eval_function(self):
        if self._can_use_gradient_deviations():
            return '''
                #ifndef GET_NEW_GRADIENT_RAW
                #define GET_NEW_GRADIENT_RAW
                mot_float_type4 _get_new_gradient_raw(mot_float_type4 g,
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
               and 'g' in list(self.get_problems_protocol_data().keys())

    def _add_finalizing_result_maps(self, results_dict):
        super(DMRISingleModel, self)._add_finalizing_result_maps(results_dict)

        log_likelihood_calc = LogLikelihoodCalculator()
        log_likelihoods = log_likelihood_calc.calculate(self, results_dict)

        k = self.get_nmr_estimable_parameters()
        n = self._problem_data.get_nmr_inst_per_problem()

        results_dict.update({'LogLikelihood': log_likelihoods})
        results_dict.update(utils.calculate_information_criterions(log_likelihoods, k, n))

    def _prepare_problem_data(self, problem_data):
        """Update the problem data to make it suitable for this model.

        Some of the models in diffusion MRI can only handle a subset of all volumes. For example, the S0 model
        can only work with the unweigthed signals, or the Tensor model that can only handle a b-value up to 1.5e9 s/m^2.

        Overwrite this function to limit the problem data to a suitable range.

        Args:
            problem_data (DMRIProblemData): the problem data set by the user

        Returns:
            DMRIProblemData: either the same problem data or a changed copy.
        """
        protocol = problem_data.protocol
        indices = self._get_suitable_volume_indices(problem_data)

        if len(indices) != protocol.length:
            self._logger.info('For this model, {}, we will use a subset of the protocol and DWI.'.format(self._name))
            self._logger.info('Using {} out of {} volumes, indices: {}'.format(
                len(indices), protocol.length, str(indices).replace('\n', '').replace('[  ', '[')))

            new_protocol = protocol.get_new_protocol_with_indices(indices)
            new_dwi_volume = problem_data.dwi_volume[..., indices]
            return problem_data.copy_with_updates(new_protocol, new_dwi_volume)
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


class DMRISingleModelConfig(ComponentConfig):
    """The cascade config to inherit from.

    These configs are loaded on the fly by the DMRISingleModelBuilder

    Attributes:
        name (str): the name of the model, defaults to the class name
        in_vivo_suitable (boolean): flag indicating if the model is suitable for in vivo data
        ex_vivo_suitable (boolean): flag indicating if the model is suitable for ex vivo data
        description (str): model description
        post_optimization_modifiers (list): a list of modification callbacks for use after optimization. Example:

            .. code-block:: python

                post_optimization_modifiers = [('SNIF', lambda d: 1 - d['Wcsf.w']),
                                           ...]
        dependencies (list): the dependencies between model parameters. Example:

            .. code-block:: python

                dependencies = [('Noddi_EC.kappa', SimpleAssignment('Noddi_IC.kappa')),
                                ...]

        model_expression (str): the model expression. For the syntax see:
            mdt.models.parsers.SingleModelExpression.ebnf
        evaluation_model (EvaluationModel): the evaluation model to use during optimization
        signal_noise_model (SignalNoiseModel): optional signal noise decorator
        inits (dict): indicating the initialization values for the parameters. Example:

            .. code-block:: python

                inits = {'Stick.theta: pi}

        fixes (dict): indicating the constant value for the given parameters. Example:

            .. code-block:: python

                fixes = {'Ball.d': 3.0e-9}

        upper_bounds (dict): indicating the upper bounds for the given parameters. Example:

            .. code-block:: python

                upper_bounds = {'Stick.theta': pi}

        lower_bounds (dict): indicating the lower bounds for the given parameters. Example:

            .. code-block:: python

                lower_bounds = {'Stick.theta': 0}

        parameter_transforms (dict): the parameter transform to use for a specific parameter. Can also be
            a python callback function accepting as single parameter 'self', a reference to the build model.
            Example:

            .. code-block:: python

                parameter_transforms = {
                    'Tensor.dperp0': SinSqrClampTransform(),
                    'Tensor.dperp1': lambda self: SinSqrClampDependentTransform(
                                                    [(self, self._get_parameter_by_name('Tensor.dperp0'))])
                }
    """
    name = ''
    in_vivo_suitable = True
    ex_vivo_suitable = True
    description = ''
    post_optimization_modifiers = []
    dependencies = []
    model_expression = ''
    evaluation_model = OffsetGaussianEvaluationModel()
    signal_noise_model = None
    inits = {}
    fixes = {}
    upper_bounds = {}
    lower_bounds = {}
    parameter_transforms = {}

    @classmethod
    def meta_info(cls):
        meta_info = deepcopy(ComponentConfig.meta_info())
        meta_info.update({'name': cls.name,
                          'in_vivo_suitable': cls.in_vivo_suitable,
                          'ex_vivo_suitable': cls.ex_vivo_suitable,
                          'description': cls.description})
        return meta_info


class DMRISingleModelBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class DMRISingleModel

        Args:
            template (DMRISingleModelConfig): the single model config template
                to use for creating the class with the right init settings.
        """
        class AutoCreatedDMRISingleModel(method_binding_meta(template, DMRISingleModel)):

            def __init__(self, *args):
                super(AutoCreatedDMRISingleModel, self).__init__(
                    deepcopy(template.name),
                    CompartmentModelTree(parse(template.model_expression)),
                    deepcopy(template.evaluation_model),
                    signal_noise_model=deepcopy(template.signal_noise_model))

                self.add_parameter_dependencies(deepcopy(template.dependencies))
                self.add_post_optimization_modifiers(deepcopy(template.post_optimization_modifiers))

                for full_param_name, value in template.inits.items():
                    self.init(full_param_name, deepcopy(value))

                for full_param_name, value in template.fixes.items():
                    self.fix(full_param_name, deepcopy(value))

                for full_param_name, value in template.lower_bounds.items():
                    self.set_lower_bound(full_param_name, deepcopy(value))

                for full_param_name, value in template.upper_bounds.items():
                    self.set_upper_bound(full_param_name, deepcopy(value))

                for full_param_name, value in template.parameter_transforms.items():
                    if hasattr(value, '__call__'):
                        self.set_parameter_transform(full_param_name, value(self))
                    else:
                        self.set_parameter_transform(full_param_name, deepcopy(value))

        return AutoCreatedDMRISingleModel
