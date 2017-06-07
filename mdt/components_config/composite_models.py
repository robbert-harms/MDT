from copy import deepcopy
import numpy as np
import six
from mdt.components_loader import ComponentConfig, ComponentBuilder, method_binding_meta, get_component_class
from mdt.models.composite import DMRICompositeModel
from mdt.models.parsers.CompositeModelExpressionParser import parse
from mot.model_building.evaluation_models import EvaluationModel
from mot.model_building.trees import CompartmentModelTree
from mot.model_building.utils import ModelPrior, SimpleModelPrior

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICompositeModelConfig(ComponentConfig):
    """The cascade config to inherit from.

    These configs are loaded on the fly by the DMRICompositeModelBuilder

    Attributes:
        name (str): the name of the model, defaults to the class name
        description (str): model description
        post_optimization_modifiers (list): a list of modification callbacks for use after optimization. Examples:

            .. code-block:: python

                post_optimization_modifiers = [('FS', lambda d: 1 - d['w_ball.w']),
                                               ('Ball.d', lambda d: d['Ball.d'] * 1e9),
                                               (['Power2', 'Power3'], lambda d: [d['foo']**2, d['foo']**3]),
                                           ...]

            The last entry in the above example shows that it is possible to include more than one
            modifier in one modifier expression.

        model_expression (str): the model expression. For the syntax see:
            mdt.models.parsers.CompositeModelExpression.ebnf
        evaluation_model (EvaluationModel or str): the evaluation model to use during optimization,
            also a string can be given with one of 'Gaussian', 'OffsetGaussian' or 'Rician'.
        signal_noise_model (SignalNoiseModel): optional signal noise decorator
        inits (dict): indicating the initialization values for the parameters. Example:

            .. code-block:: python

                inits = {'Stick.theta': np.pi}

        fixes (dict): indicating the constant value for the given parameters. Example:

            .. code-block:: python

                fixes = {'Ball.d': 3.0e-9,
                         'NODDI_EC.kappa': SimpleAssignment('NODDI_IC.kappa'),
                         'NODDI_EC.theta': 'NODDI_IC.theta'}

            Next to values, this also accepts strings as dependencies (or dependecy objects directly).

        upper_bounds (dict): indicating the upper bounds for the given parameters. Example:

            .. code-block:: python

                upper_bounds = {'Stick.theta': pi}

        lower_bounds (dict): indicating the lower bounds for the given parameters. Example:

            .. code-block:: python

                lower_bounds = {'Stick.theta': 0}

        enforce_weights_sum_to_one (boolean): set to False to disable the automatic Weight-sum-to-one dependency.
            By default it is True and we add them.

        volume_selection (dict): the volume selection by this model. This can be used to limit the volumes used
            in the analysis to only the volumes included in the specification.
            Set to None, or an empty dict to disable.
            The options available are:

               * ``unweighted_threshold`` (float): the threshold differentiating between
                 weighted and unweighted volumes
               * ``use_unweighted`` (bool): if we want to use unweighted volumes or not
               * ``use_weighted`` (bool): if we want to use the diffusion weigthed volumes or not
               * ``min_bval`` (float): the minimum b-value to include
               * ``max_bval`` (float): the maximum b-value to include

            If the method ``_get_suitable_volume_indices`` is overwritten, this does nothing.

        prior (str, mot.model_building.utils.ModelPrior or None): a model wide prior. This is used in conjunction with
            the compartment priors and the parameter priors. If a string is given we will automatically construct a
            :class:`mot.model_building.utils.ModelPrior` from that string.
    """
    name = ''
    description = ''
    post_optimization_modifiers = []
    model_expression = ''
    evaluation_model = 'OffsetGaussian'
    signal_noise_model = None
    inits = {}
    fixes = {}
    upper_bounds = {}
    lower_bounds = {}
    enforce_weights_sum_to_one = True
    volume_selection = None
    prior = None

    @classmethod
    def meta_info(cls):
        meta_info = deepcopy(ComponentConfig.meta_info())
        meta_info.update({'name': cls.name,
                          'description': cls.description})
        return meta_info


class DMRICompositeModelBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class DMRICompositeModel

        Args:
            template (DMRICompositeModelConfig): the composite model config template
                to use for creating the class with the right init settings.
        """
        class AutoCreatedDMRICompositeModel(method_binding_meta(template, DMRICompositeModel)):

            def __init__(self):
                super(AutoCreatedDMRICompositeModel, self).__init__(
                    deepcopy(template.name),
                    CompartmentModelTree(parse(template.model_expression)),
                    deepcopy(_resolve_evaluation_model(template.evaluation_model)),
                    signal_noise_model=deepcopy(template.signal_noise_model),
                    enforce_weights_sum_to_one=template.enforce_weights_sum_to_one)

                self.add_post_optimization_modifiers(_get_model_post_optimization_modifiers(
                    self._model_functions_info.get_model_list()))
                self.add_post_optimization_modifiers(deepcopy(template.post_optimization_modifiers))

                for full_param_name, value in template.inits.items():
                    self.init(full_param_name, deepcopy(value))

                for full_param_name, value in template.fixes.items():
                    self.fix(full_param_name, deepcopy(value))

                for full_param_name, value in template.lower_bounds.items():
                    self.set_lower_bound(full_param_name, deepcopy(value))

                for full_param_name, value in template.upper_bounds.items():
                    self.set_upper_bound(full_param_name, deepcopy(value))

                self.nmr_parameters_for_bic_calculation = self.get_nmr_estimable_parameters()

                self._model_priors.extend(_resolve_model_prior(
                    template.prior, self._model_functions_info.get_model_parameter_list()))

            def _get_suitable_volume_indices(self, problem_data):
                volume_selection = template.volume_selection

                if not volume_selection:
                    return super(AutoCreatedDMRICompositeModel, self)._get_suitable_volume_indices(problem_data)

                use_unweighted = volume_selection.get('use_unweighted', True)
                use_weighted = volume_selection.get('use_weighted', True)
                unweighted_threshold = volume_selection.get('unweighted_threshold', 25e6)

                protocol = problem_data.protocol

                if protocol.has_column('g') and protocol.has_column('b'):
                    if use_weighted:
                        if 'min_bval' in volume_selection and 'max_bval' in volume_selection:
                            protocol_indices = protocol.get_indices_bval_in_range(start=volume_selection['min_bval'],
                                                                                  end=volume_selection['max_bval'])
                        else:
                            protocol_indices = protocol.get_weighted_indices(unweighted_threshold)
                    else:
                        protocol_indices = []

                    if use_unweighted:
                        protocol_indices = list(protocol_indices) + \
                                           list(protocol.get_unweighted_indices(unweighted_threshold))
                else:
                    return list(range(protocol.length))

                return np.unique(protocol_indices)

        return AutoCreatedDMRICompositeModel


def _resolve_evaluation_model(evaluation_model):
    """Resolve the evaluation model from string if necessary.

    The composite models accept evaluation models from string and evaluation models as object. This function
    resolves the strings if a string is given, else it returns the object passed.

    Args:
        evaluation_model (str or object): the evaluation model to resolve to an object

    Returns:
        mot.model_building.evaluation_models.EvaluationModel: the evaluation model to use
    """
    if isinstance(evaluation_model, six.string_types):
        return get_component_class('evaluation_models', evaluation_model + 'EvaluationModel')()
    else:
        return evaluation_model


def _resolve_model_prior(prior, model_parameters):
    """Resolve the model priors.

    Args:
        model_prior (None or str or mot.model_building.utils.ModelPrior): the prior defined in the composite model
            template.
        model_parameters (str): the (model, parameter) tuple for all the parameters in the model

    Returns:
        list of mot.model_building.utils.ModelPrior: list of model priors
    """
    if prior is None:
        return []

    if isinstance(prior, ModelPrior):
        return [prior]

    parameters = []
    for m, p in model_parameters:
        dotted_name = '{}.{}'.format(m.name, p.name)
        bar_name = dotted_name.replace('.', '_')

        if dotted_name in prior:
            prior = prior.replace(dotted_name, bar_name)
            parameters.append(dotted_name)
        elif bar_name in prior:
            parameters.append(dotted_name)

    return [SimpleModelPrior(prior, parameters, 'model_prior')]


def _get_model_post_optimization_modifiers(compartments):
    """Get a list of all the post optimization modifiers defined in the models.

    This function will add a wrapper around the modification routines to make the input and output maps relative to the
    model. That it, these functions expect the parameter names without the model name and output map names without
    the model name, whereas the expected input and output of the modifiers of the model is with the full model.map name.

    Args:
        compartments (list): the list of compartment models from which to get the modifiers
    """
    modifiers = []

    def get_wrapped_modifier(compartment_name, original_map_names, original_modifier):
        single_output = isinstance(original_map_names, six.string_types)

        if single_output:
            wrapped_names = '{}.{}'.format(compartment_name, original_map_names)
        else:
            wrapped_names = ['{}.{}'.format(compartment_name, map_name) for map_name in original_map_names]

        def wrapped_modifier(maps):
            compartment_specific_maps = {k[len(compartment_name) + 1:]: v for k, v in maps.items()
                                         if k.startswith(compartment_name)}
            return original_modifier(compartment_specific_maps)

        return wrapped_names, wrapped_modifier

    for compartment in compartments:
        if hasattr(compartment, 'post_optimization_modifiers'):
            for map_names, modifier in compartment.post_optimization_modifiers:
                modifiers.append(get_wrapped_modifier(compartment.name, map_names, modifier))

    return modifiers
