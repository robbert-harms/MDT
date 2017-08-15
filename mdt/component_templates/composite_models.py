import inspect
from copy import deepcopy
import numpy as np
import six
from mdt.components_loader import get_component_class
from mdt.component_templates.base import ComponentBuilder, method_binding_meta, ComponentTemplate, register_builder
from mdt.models.composite import DMRICompositeModel
from mdt.models.parsers.CompositeModelExpressionParser import parse
from mot.model_building.evaluation_models import EvaluationModel
from mot.model_building.trees import CompartmentModelTree
from mot.model_building.utils import ModelPrior, SimpleModelPrior


__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICompositeModelTemplate(ComponentTemplate):
    """The cascade config to inherit from.

    These configs are loaded on the fly by the DMRICompositeModelBuilder

    Attributes:
        name (str): the name of the model, defaults to the class name

        description (str): model description

        post_optimization_modifiers (list): a list of modification callbacks for use after optimization. Examples:

            .. code-block:: python

                post_optimization_modifiers = [('FS', lambda d: 1 - d['w_ball.w']),
                                               ('Kurtosis.MK', lambda d, protocol: <...>),
                                               (['Power2', 'Power3'], lambda d: [d['foo']**2, d['foo']**3]),
                                           ...]

            The last entry in the above example shows that it is possible to include more than one
            modifier in one modifier expression. In general, the function given should accept as first argument
            the results dictionary and as optional second argument the protocol used to generate the results.
            These modifiers are called after the modifiers of the composite model.

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

        sort_maps (list of tuple): The maps to sort as post-processing before the other post optimization modifiers.
            This will sort the given maps voxel by voxel based on the given maps as a key. The first
            tuple needs to be a parameter reference or a ``Weight`` compartment, the other tuples can contain either
            parameters or compartments.
            Example input::

                maps_to_sort = [('w0.w', 'w1.w'), ('Stick0', 'Stick1')]

            will sort the weights w0.w and w1.w and sort all the parameters of the Stick0 and Stick1
            compartment according to the sorting of those weights.

            One can also sort on only one parameter of a compartment using::

                maps_to_sort = [('w0', 'w1'), ('Stick0.theta', 'Stick1.theta')]

            which will again sort the weights but will only sort the theta map of both the Stick compartments.
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
    extra_prior = None
    sort_maps = None

    @classmethod
    def meta_info(cls):
        meta_info = deepcopy(ComponentTemplate.meta_info())
        meta_info.update({'name': cls.name,
                          'description': cls.description})
        return meta_info


class DMRICompositeModelBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class DMRICompositeModel

        Args:
            template (DMRICompositeModelTemplate): the composite model config template
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

                if template.sort_maps:
                    self._post_optimization_modifiers.append(_get_map_sorting_modifier(
                        template.sort_maps, self._model_functions_info.get_model_parameter_list()))

                self._post_optimization_modifiers.extend(_get_model_post_optimization_modifiers(
                    self._model_functions_info.get_model_list()))
                self._post_optimization_modifiers.extend(deepcopy(template.post_optimization_modifiers))

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
                    template.extra_prior, self._model_functions_info.get_model_parameter_list()))

                self._sampling_covar_extras.extend(_get_model_sampling_covariance_extras(
                    self._model_functions_info.get_model_list()))
                self._sampling_covar_excludes.extend(_get_model_sampling_covariance_excludes(
                    self._model_functions_info.get_model_list()))

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


def _get_map_sorting_modifier(sort_maps, model_list):
    """Construct the map sorting modification routine for the given maps_to_sort attribute.

    Args:
        sort_maps (list of tuple): the list with compartments/parameters to sort. Sorting is done on the first element and
            the other maps are sorted based on that indexing.
        model_list (list of tuple): the list of model, parameter tuples to use for matching names.
    Returns:
        tuple: name, modification routine. The modification routine will sort the maps as specified.
    """
    def is_param_name(name):
        return '.' in name

    def get_model_by_name(name):
        for model, parameter in model_list:
            if model.name == name:
                return model

    sort_keys = []
    for name in sort_maps[0]:
        for model, parameter in model_list:
            full_name = '{}.{}'.format(model.name, parameter.name)
            if model.name == name or full_name == name:
                sort_keys.append(full_name)
                break

    other_sort_pairs = []
    for other_pair in sort_maps[1:]:
        if is_param_name(other_pair[0]):
            other_sort_pairs.append(other_pair)
        else:
            model = get_model_by_name(other_pair[0])
            for parameter in model.get_free_parameters():
                sort_pair = []
                for model_name in other_pair:
                    sort_pair.append('{}.{}'.format(model_name, parameter.name))
                other_sort_pairs.append(sort_pair)

    def map_sorting(results):
        sort_matrix = np.column_stack([results[map_name] for map_name in sort_keys])
        ranking = np.atleast_2d(np.squeeze(np.argsort(sort_matrix, axis=1)[:, ::-1]))
        list_index = np.arange(ranking.shape[0])

        sorted_maps = []
        sorted_maps.extend(sort_matrix[list_index, ranking[:, ind], None] for ind in range(ranking.shape[1]))

        for pair in other_sort_pairs:
            sort_matrix = np.column_stack([results[map_name] for map_name in pair])
            sorted_maps.extend(sort_matrix[list_index, ranking[:, ind], None] for ind in range(ranking.shape[1]))
        return sorted_maps

    names = []
    names.extend(sort_keys)
    for pairs in other_sort_pairs:
        names.extend(pairs)

    return names, map_sorting


def _get_model_post_optimization_modifiers(compartments):
    """Get a list of all the post optimization modifiers defined in the models.

    This function will add a wrapper around the modification routines to make the input and output maps relative to the
    model. That it, these functions expect the parameter names without the model name and output map names without
    the model name, whereas the expected input and output of the modifiers of the model is with the full model.map name.

    Args:
        compartments (list): the list of compartment models from which to get the modifiers

    Returns:
        list of tuples: the list of modification names and routines. Example:
            [('name1', mod1), ('name2', mod2), ...]
    """
    modifiers = []

    def get_wrapped_modifier(compartment_name, original_map_names, original_modifier):
        single_output = isinstance(original_map_names, six.string_types)

        if single_output:
            wrapped_names = '{}.{}'.format(compartment_name, original_map_names)
        else:
            wrapped_names = ['{}.{}'.format(compartment_name, map_name) for map_name in original_map_names]

        def get_compartment_specific_names(results):
            return {k[len(compartment_name) + 1:]: v for k, v in results.items() if k.startswith(compartment_name)}

        argspec = inspect.getfullargspec(original_modifier)
        if len(argspec.args) > 1:
            def wrapped_modifier(results, protocol):
                return original_modifier(get_compartment_specific_names(results), protocol)
        else:
            def wrapped_modifier(results):
                return original_modifier(get_compartment_specific_names(results))

        return wrapped_names, wrapped_modifier

    for compartment in compartments:
        if hasattr(compartment, 'post_optimization_modifiers'):
            for map_names, modifier in compartment.post_optimization_modifiers:
                modifiers.append(get_wrapped_modifier(compartment.name, map_names, modifier))

    return modifiers


def _get_model_sampling_covariance_excludes(compartments):
    """Get a list of the model parameters to remove before calculating the sampling covariance.

    Args:
        compartments (list): the list of compartment models from which to get the modifiers

    Returns:
        list: The list of full model names we need to remove
    """
    excludes = []

    for compartment in compartments:
        if hasattr(compartment, 'sampling_covar_exclude') and compartment.sampling_covar_exclude is not None \
                and len(compartment.sampling_covar_exclude):
            for param_name in compartment.sampling_covar_exclude:
                excludes.append('{}.{}'.format(compartment.name, param_name))

    return excludes

def _get_model_sampling_covariance_extras(compartments):
    """Get a list with the information about the additional maps to include in the covariance calculation.

    Args:
        compartments (list): the list of compartment models from which to get the modifiers

    Returns:
        list: list with tuple of (list, list, Func), information about the extra maps to include
    """
    extras = []

    for compartment in compartments:
        if hasattr(compartment, 'sampling_covar_extras') and compartment.sampling_covar_extras is not None \
                and len(compartment.sampling_covar_extras):

            for input_params, output_params, func in compartment.sampling_covar_extras:
                input_params = ['{}.{}'.format(compartment.name, p) for p in input_params]
                output_params = ['{}.{}'.format(compartment.name, p) for p in output_params]
                extras.append((input_params, output_params, func))

    return extras


register_builder(DMRICompositeModelTemplate, DMRICompositeModelBuilder())
