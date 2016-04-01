import logging
from copy import deepcopy
import numpy as np
from mdt import utils
from mdt.components_loader import ComponentConfig, ComponentBuilder
from mdt.models.base import DMRIOptimizable
from mdt.models.parsers.SingleModelExpressionParser import parse
from mot.adapters import SimpleDataAdapter
from mot.base import CLDataType
from mot.cl_functions import Weight
from mdt.utils import restore_volumes, create_roi
from mdt.model_protocol_problem import MissingColumns, InsufficientShells
from mot.cl_routines.mapping.loglikelihood_calculator import LogLikelihoodCalculator
from mot.model_building.evaluation_models import GaussianEvaluationModel, OffsetGaussianEvaluationModel
from mot.model_building.parameter_functions.dependencies import WeightSumToOneRule
from mot.models import SmoothableModelInterface, PerturbationModelInterface
from mot.model_building.model_builders import SampleModelBuilder
from mot.trees import CompartmentModelTree

__author__ = 'Robbert Harms'
__date__ = "2014-10-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRISingleModel(SampleModelBuilder, SmoothableModelInterface, DMRIOptimizable, PerturbationModelInterface):

    def __init__(self, model_name, model_tree, evaluation_model, signal_noise_model=None, problem_data=None):
        """Create a composite dMRI sample model.

        This also implements the smoothing interface to allow spatial smoothing of the data during meta-optimization.

        It furthermore implements some protocol check functions. These are used by the fit_model functions in MDT
        to check if the protocol is correct for the model we try to fit.

        Attributes:
            white_list (list): The list of names of maps that must be smoothed. Set to None to ignore, set to [] to
                filter no maps.
            black_list (list): The list of names of maps that must not be smoothed. Set to None to ignore, set to [] to
                allow all maps.
            required_nmr_shells (int): Define the minimum number of unique shells necessary for this model.
                The default is false, which means that we don't check for this.
            grad_dev (ndarray): contains the effects of gradient nonlinearities on the bvals and bvecs for each voxel.
                This should be a 2d matrix with per voxel 9 values that constitute the gradient deviation in
                Fortran (column-major) order. This data is used as defined by the HCP WUMINN study.
        """
        super(DMRISingleModel, self).__init__(model_name, model_tree, evaluation_model, signal_noise_model,
                                              problem_data=problem_data)
        self.smooth_white_list = None
        self.smooth_black_list = None
        self.required_nmr_shells = False
        self.gradient_deviations = None
        self._logger = logging.getLogger(__name__)

    @property
    def evaluation_model(self):
        return self._evaluation_model

    def set_smooth_lists(self, white_list=None, black_list=None):
        """Set the list with maps to filter.

        If the white list is set the black list is ignored. If the white list is not set and the black list is set
        then the black list is used.

        The white list regulates the maps that must be smoothed. The black list lists those that must be ignored.

        Args:
            white_list (list): The list of names of maps that must be smoothed. Set to None to ignore, set to [] to
                filter no maps.
            black_list (list): The list of names of maps that must not be smoothed. Set to None to ignore, set to [] to
                allow all maps.
        """
        self.smooth_white_list = white_list
        self.smooth_black_list = black_list

    def set_gradient_deviations(self, grad_dev):
        """Set the gradient deviations.

        Args:
            grad_dev (ndarray): the gradient deviations containing per voxel 9 values that constitute the gradient
                non-linearities. The matrix can either be a 4d matrix or a 2d matrix.

                If it is a 4d matrix the first 3 dimensions are supposed to be the voxel index and the 4th
                should contain the grad dev data.

                If it is a 2 dimensional matrix, the first dimension is the voxel index and the second should contain
                the gradient deviation data. In this case the nmr of voxels should coincide with the number of voxels
                in the ROI of the DWI.
        """
        self.gradient_deviations = grad_dev
        return self

    def get_problems_var_data(self):
        var_data_dict = super(DMRISingleModel, self).get_problems_var_data()

        if self.gradient_deviations is not None:
            if len(self.gradient_deviations.shape) > 2:
                grad_dev = create_roi(self.gradient_deviations, self._problem_data.mask)
            else:
                grad_dev = np.copy(self.gradient_deviations)

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

    def smooth(self, results, smoother):
        if self.smooth_white_list is not None:
            smoothable = [e for e in self.get_optimized_param_names() if e in self.smooth_white_list]
        elif self.smooth_black_list is not None:
            smoothable = [e for e in self.get_optimized_param_names() if e not in self.smooth_black_list]
        else:
            smoothable = self.get_optimized_param_names()

        if smoothable:
            mask = self._problem_data.mask
            to_smooth = {key: value for key, value in results.items() if key in smoothable}
            not_to_smooth = {key: value for key, value in results.items() if key not in smoothable}

            roi_smoothed = smoother.filter(restore_volumes(to_smooth, mask, with_volume_dim=False), mask,
                                           self.double_precision)
            smoothed_results = create_roi(roi_smoothed, mask)
            smoothed_results.update(not_to_smooth)
            return smoothed_results
        return results

    def perturbate(self, results):
        for map_name in self.get_optimized_param_names():
            if map_name in results:
                param = self._get_parameter_by_name(map_name)
                results[map_name] = param.perturbation_function(results[map_name])
        return results

    def is_protocol_sufficient(self, protocol=None):
        """See ProtocolCheckInterface"""
        return not self.get_protocol_problems(protocol=protocol)

    def get_protocol_problems(self, protocol=None):
        """See ProtocolCheckInterface"""
        if protocol is None:
            protocol = self._problem_data.protocol_data_dict

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
        return self.gradient_deviations is not None and 'g' in list(self.get_problems_protocol_data().keys())

    def _add_finalizing_result_maps(self, results_dict):
        log_likelihood_calc = LogLikelihoodCalculator()
        log_likelihoods = log_likelihood_calc.calculate(self, results_dict)

        k = self.get_nmr_estimable_parameters()
        n = self._problem_data.protocol.length

        results_dict.update({'LogLikelihood': log_likelihoods})
        results_dict.update(utils.calculate_information_criterions(log_likelihoods, k, n))


class DMRISingleModelConfig(ComponentConfig):
    """The cascade config to inherit from.

    These configs are loaded on the fly by the DMRISingleModelBuilder

    Config options:
        name (str): the name of the model
        in_vivo_suitable (boolean): flag indicating if the model is suitable for in vivo data
        ex_vivo_suitable (boolean): flag indicating if the model is suitable for ex vivo data
        description (str): model description
        post_optimization_modifiers (list): a list of modification callbacks for use after optimization. Example:
            post_optimization_modifiers = [('SNIF', lambda d: 1 - d['Wcsf.w']),
                                           ...]
        dependencies (list): the dependencies between model parameters. Example:
            dependencies = [('Noddi_EC.kappa', SimpleAssignment('Noddi_IC.kappa')),
                            ...]
        model_expression (str): the model expression. For the syntax see
            mdt.models.parsers.SingleModelExpression.ebnf
        evaluation_model (EvaluationModel): the evaluation model to use during optimization
        signal_noise_model (SignalNoiseModel): optional signal noise decorator
        inits (dict): indicating the initialization values for the parameters. Example:
            inits = {'Stick.theta: pi}
        fixes (dict): indicating the constant value for the given parameters. Example:
            fixes = {'Ball.d': 3.0e-9}
        upper_bounds (dict): indicating the upper bounds for the given parameters. Example:
            upper_bounds = {'Stick.theta': pi}
        lower_bounds (dict): indicating the lower bounds for the given parameters. Example:
            lower_bounds = {'Stick.theta': 0}
        parameter_transforms (dict): the parameter transform to use for a specific parameter. Can also be
            a python callback function accepting as single parameter 'self', a reference to the build model.
            Example:
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
    evaluation_model = OffsetGaussianEvaluationModel().fix('sigma', 1)
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
        class AutoCreatedDMRISingleModel(DMRISingleModel):

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

        self._bind_functions(template, AutoCreatedDMRISingleModel)
        return AutoCreatedDMRISingleModel
