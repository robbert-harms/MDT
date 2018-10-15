import inspect
import re
from copy import deepcopy
import numpy as np
import tatsu

from mdt.component_templates.base import ComponentBuilder, ComponentTemplate
from mdt.lib.components import get_component
from mdt.models.composite import DMRICompositeModel
from mot.lib.cl_function import CLFunction, SimpleCLFunction
from mdt.model_building.trees import CompartmentModelTree
import collections

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


_composite_model_expression_parser = tatsu.compile('''
    result = expr;
    expr = term ('+'|'-') expr | term;
    term = factor ('*'|'/') term | factor;
    factor = '(' expr ')' | model;
    model = model_name ['(' nickname ')'];
    model_name = /[a-zA-Z_]\w*/;
    nickname = /[a-zA-Z_]\w*/;
''')


class DMRICompositeModelBuilder(ComponentBuilder):

    def _create_class(self, template):
        """Creates classes with as base class DMRICompositeModel

        Args:
            template (CompositeModelTemplate): the composite model config template
                to use for creating the class with the right init settings.
        """
        class AutoCreatedDMRICompositeModel(DMRICompositeModel):

            def __init__(self, volume_selection=True):
                super().__init__(
                    deepcopy(template.name),
                    CompartmentModelTree(parse_composite_model_expression(template.model_expression)),
                    deepcopy(_resolve_likelihood_function(template.likelihood_function)),
                    signal_noise_model=deepcopy(template.signal_noise_model),
                    enforce_weights_sum_to_one=template.enforce_weights_sum_to_one,
                    volume_selection=volume_selection
                )

                for full_param_name, value in template.inits.items():
                    self.init(full_param_name, deepcopy(value))

                for full_param_name, value in template.fixes.items():
                    self.fix(full_param_name, deepcopy(value))

                for full_param_name, value in template.lower_bounds.items():
                    self.set_lower_bound(full_param_name, deepcopy(value))

                for full_param_name, value in template.upper_bounds.items():
                    self.set_upper_bound(full_param_name, deepcopy(value))

                self.nmr_parameters_for_bic_calculation = self.get_nmr_parameters()

                if template.sort_maps:
                    self._post_optimization_modifiers.append(_get_map_sorting_modifier(
                        template.sort_maps, self._model_functions_info.get_model_parameter_list()))

                self._post_optimization_modifiers.extend(_get_model_post_optimization_modifiers(
                    self._model_functions_info.get_compartment_models()))
                self._post_optimization_modifiers.extend(deepcopy(template.post_optimization_modifiers))

                self._extra_optimization_maps_funcs.extend(_get_model_extra_optimization_maps_funcs(
                    self._model_functions_info.get_compartment_models()))
                self._extra_optimization_maps_funcs.extend(deepcopy(template.extra_optimization_maps))

                self._extra_sampling_maps_funcs.extend(_get_model_extra_sampling_maps_funcs(
                    self._model_functions_info.get_compartment_models()))
                self._extra_sampling_maps_funcs.extend(deepcopy(template.extra_sampling_maps))

                self._proposal_callbacks.extend(_get_model_proposal_callbacks(
                    self._model_functions_info.get_compartment_models()))

                self._model_priors.extend(_resolve_model_prior(
                    template.extra_prior, self._model_functions_info.get_model_parameter_list()))

            def _get_suitable_volume_indices(self, input_data):
                volume_selection = template.volume_selection

                if not volume_selection:
                    return super()._get_suitable_volume_indices(input_data)

                use_unweighted = volume_selection.get('use_unweighted', True)
                use_weighted = volume_selection.get('use_weighted', True)
                unweighted_threshold = volume_selection.get('unweighted_threshold', 25e6)

                protocol = input_data.protocol

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

        for name, method in template.bound_methods.items():
            setattr(AutoCreatedDMRICompositeModel, name, method)

        return AutoCreatedDMRICompositeModel


class CompositeModelTemplate(ComponentTemplate):
    """The cascade config to inherit from.

    These configs are loaded on the fly by the DMRICompositeModelBuilder

    Attributes:
        name (str): the name of the model, defaults to the class name

        description (str): model description

        post_optimization_modifiers (list): a list of modification callbacks to change the estimated
            optimization points. This should not add new maps, for that use the ``additional_results`` directive.
            This can be used to, for example, sort maps into a similar valued but sorted representation
            (actually the ``sort_maps`` directive creates a post_optimization_modifier)

            Examples:

            .. code-block:: python

                post_optimization_modifiers = [lambda d: reorient_tensor(d),
                                               lambda d: {'w_stick0.w': ..., 'w_stick1.w': ...}
                                                ...]

            This should return a dictionary with updated maps.

        sort_maps (list of tuple): The maps to sort voxel-by voxel as post-processing.
            To ensure that the point estimate from optimization can directly be used in MCMC sample, we sometimes
            need to rearrange the weights (and therefore the corresponding compartments). For example, some composite
            models have a prior on the weights of similar compartments to restrict them to a decreasing order, which
            is typically done to prevent bimodal continuous_distributions. This model directive allows you to easily specify
            which parameters to rearrange as first post-processing of optimization results.

            The first tuple needs to be a parameter reference or a  ``Weight`` compartment, the other tuples can
            contain either parameters or compartments.

            Example input::

                sort_maps = [('w0.w', 'w1.w'), ('Stick0', 'Stick1')]

            this input sorts the weights w0.w and w1.w and afterwards sorts all the parameters of the Stick0 and Stick1
            compartment according to the sorting of those weights.

            One can also sort just one parameter of a compartment using::

                sort_maps = [('w0', 'w1'), ('Stick0.theta', 'Stick1.theta')]

            which will again sort the weights but will only apply the sorting on the theta map of both the Stick
            compartments.

        extra_optimization_maps (list): a list of functions to return extra information maps based on a point estimate.
            This is called after the post optimization modifiers and after the model calculated uncertainties based
            on the Fisher Information Matrix. Therefore, these routines can propagate uncertainties in the estimates.
            Please note that this is only for adding additional maps. For changing the point estimate of the
            optimization, please use the ``post_optimization_modifiers`` directive.

            These functions should accept as single argument an object of type
            :class:`mdt.models.composite.ExtraOptimizationMapsInfo`.

            Examples::

                extra_optimization_maps = [lambda d: {'FS': 1 - d['w_ball.w']},
                                           lambda d: {'Kurtosis.MK': <...>},
                                           lambda d: {'Power2': d['foo']**2, 'Power3': d['foo']**3},
                                           ...]

        extra_sampling_maps (list): a list of functions to return additional maps as results from sample.
            This is called after sample with as argument a dictionary containing the sample results and
            the values of the fixed parameters.

            Examples::

                extra_sampling_maps = [lambda d: {'FS': np.mean(d['w_stick0.w'], axis=1),
                                                  'FS.std': np.std(d['w_stick0.w'], axis=1)}
                                      ...]

        model_expression (str): the model expression. For the syntax see:
            mdt.models.parsers.CompositeModelExpression.ebnf

        likelihood_function (:class:`mdt.model_building.likelihood_functions.LikelihoodFunction` or str): the
            likelihood function to use during optimization, can also can be a string with one of
            'Gaussian', 'OffsetGaussian' or 'Rician'

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

        extra_prior (str, mdt.model_building.utils.ModelPrior or None): a model wide prior. This is used in conjunction
            with the compartment priors and the parameter priors. If a string is given we will automatically construct a
            :class:`mdt.model_building.utils.ModelPrior` from that string.
    """
    _component_type = 'composite_models'
    _builder = DMRICompositeModelBuilder()

    name = ''
    description = ''
    post_optimization_modifiers = []
    extra_optimization_maps = []
    extra_sampling_maps = []
    model_expression = ''
    likelihood_function = 'OffsetGaussian'
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


def _resolve_likelihood_function(likelihood_function):
    """Resolve the likelihood function from string if necessary.

    The composite models accept likelihood functions as a string and as a object. This function
    resolves the strings if a string is given, else it returns the object passed.

    Args:
        likelihood_function (str or object): the likelihood function to resolve to an object

    Returns:
        mdt.model_building.likelihood_models.LikelihoodFunction: the likelihood function to use
    """
    if isinstance(likelihood_function, str):
        return get_component('likelihood_functions', likelihood_function + 'LikelihoodFunction')()
    else:
        return likelihood_function


def _resolve_model_prior(prior, model_parameters):
    """Resolve the model priors.

    Args:
        prior (None or str or mot.lib.cl_function.CLFunction): the prior defined in the composite model template.
        model_parameters (str): the (model, parameter) tuple for all the parameters in the model

    Returns:
        list of mdt.model_building.utils.ModelPrior: list of model priors
    """
    if prior is None:
        return []

    if isinstance(prior, CLFunction):
        return [prior]

    dotted_names = ['{}.{}'.format(m.name, p.name) for m, p in model_parameters]
    dotted_names.sort(key=len, reverse=True)

    parameters = []
    remaining_prior = prior
    for dotted_name in dotted_names:
        bar_name = dotted_name.replace('.', '_')

        if dotted_name in remaining_prior:
            prior = prior.replace(dotted_name, bar_name)
            remaining_prior = remaining_prior.replace(dotted_name, '')
            parameters.append(('mot_float_type', dotted_name))
        elif bar_name in remaining_prior:
            remaining_prior = remaining_prior.replace(bar_name, '')
            parameters.append(('mot_float_type', dotted_name))

    return [SimpleCLFunction('mot_float_type', 'model_prior', parameters, prior)]


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

    names = []
    names.extend(sort_keys)
    for pairs in other_sort_pairs:
        names.extend(pairs)

    def map_sorting(results):
        sort_matrix = np.column_stack([results[map_name] for map_name in sort_keys])
        ranking = np.atleast_2d(np.squeeze(np.argsort(sort_matrix, axis=1)[:, ::-1]))
        list_index = np.arange(ranking.shape[0])

        sorted_maps = []
        sorted_maps.extend(sort_matrix[list_index, ranking[:, ind], None] for ind in range(ranking.shape[1]))

        for pair in other_sort_pairs:
            sort_matrix = np.column_stack([results[map_name] for map_name in pair])
            sorted_maps.extend(sort_matrix[list_index, ranking[:, ind], None] for ind in range(ranking.shape[1]))

        return dict(zip(names, sorted_maps))

    return map_sorting


def _get_model_post_optimization_modifiers(compartments):
    """Get a list of all the post optimization modifiers defined in the models.

    This function will add a wrapper around the modification routines to make the input and output maps relative to the
    model. That it, these functions expect the parameter names without the model name and output map names without
    the model name, whereas the expected input and output of the modifiers of the model is with the full model.map name.

    Args:
        compartments (list): the list of compartment models from which to get the modifiers

    Returns:
        list: the list of modification routines taken from the compartment models.
    """
    modifiers = []

    def get_wrapped_modifier(compartment_name, original_modifier):
        def get_compartment_specific_maps(results):
            return {k[len(compartment_name) + 1:]: v for k, v in results.items() if k.startswith(compartment_name)}

        def prepend_compartment_name(results):
            return {'{}.{}'.format(compartment_name, key): value for key, value in results.items()}

        def wrapped_modifier(results):
            return prepend_compartment_name(original_modifier(get_compartment_specific_maps(results)))

        return wrapped_modifier

    for compartment in compartments:
        for modifier in compartment.get_post_optimization_modifiers():
            modifiers.append(get_wrapped_modifier(compartment.name, modifier))

    return modifiers


def _get_model_extra_optimization_maps_funcs(compartments):
    """Get a list of all the additional result functions defined in the compartments.

    This function will add a wrapper around the modification routines to make the input and output maps relative to the
    model. That it, the functions in the compartments expect the parameter names without the model name and they output
    maps without the model name, whereas the expected input and output of the modifiers of the model is with the
    full model.map name.

    Args:
        compartments (list): the list of compartment models from which to get the modifiers

    Returns:
        list: the list of modification routines taken from the compartment models.
    """
    funcs = []

    def get_wrapped_func(compartment_name, original_func):
        def get_compartment_specific_results(results):
            maps = {k[len(compartment_name) + 1:]: v for k, v in results.items() if k.startswith(compartment_name)}

            if 'covariances' in results and results['covariances'] is not None:
                p = re.compile(compartment_name + r'\.\w+_to_' + compartment_name + r'\.\w+')
                maps['covariances'] = {k.replace(compartment_name + '.', ''): v
                                       for k, v in results['covariances'].items() if p.match(k)}

            return results.copy_with_different_results(maps)

        def prepend_compartment_name(results):
            return {'{}.{}'.format(compartment_name, key): value for key, value in results.items()}

        def wrapped_modifier(results):
            return prepend_compartment_name(original_func(get_compartment_specific_results(results)))

        return wrapped_modifier

    for compartment in compartments:
        for func in compartment.get_extra_optimization_maps_funcs():
            funcs.append(get_wrapped_func(compartment.name, func))

    return funcs


def _get_model_extra_sampling_maps_funcs(compartments):
    """Get a list of all the additional post-sample functions defined in the compartments.

    This function will add a wrapper around the modification routines to make the input and output maps relative to the
    model. That it, the functions in the compartments expect the parameter names without the model name and they output
    maps without the model name, whereas the expected input and output of the modifiers of the model is with the
    full model.map name.

    Args:
        compartments (list): the list of compartment models from which to get the modifiers

    Returns:
        list: the list of extra sample routines taken from the compartment models.
    """
    funcs = []

    def get_wrapped_func(compartment_name, original_func):
        def prepend_compartment_name(results):
            return {'{}.{}'.format(compartment_name, key): value for key, value in results.items()}

        def wrapped_modifier(results):
            return prepend_compartment_name(original_func(CompartmentContextResults(compartment_name, results)))

        return wrapped_modifier

    for compartment in compartments:
        for func in compartment.get_extra_sampling_maps_funcs():
            funcs.append(get_wrapped_func(compartment.name, func))

    return funcs


def _get_model_proposal_callbacks(compartments):
    """Get a list of all the additional proposal callback functions defined in the compartments.

    This function will add a wrapper around the proposal callback to make the input and output maps relative to the
    model.

    Args:
        compartments (list): the list of compartment models from which to get the modifiers

    Returns:
        list: the list of proposal callbacks taken from the compartment models.
    """
    funcs = []
    for compartment in compartments:
        for params, func in compartment.get_proposal_callbacks():
            compartment_params = [(compartment, p) for p in params]
            funcs.append((compartment_params, func))
    return funcs


class CompartmentContextResults(collections.Mapping):

    def __init__(self, compartment_name, input_results):
        """Translates the original results to the context of a single compartment.

        This basically adds a wrapper around the input dictionary to make the keys relative to the compartment.

        Args:
            compartment_name (str): the name of the compartment we are making things relative for
            input_results (dict): the original input we want to make relative
        """
        self._compartment_name = compartment_name
        self._input_results = input_results
        self._valid_keys = [key for key in self._input_results if key.startswith(self._compartment_name + '.')]

    def __getitem__(self, key):
        return self._input_results['{}.{}'.format(self._compartment_name, key)]

    def __len__(self):
        return len(self._valid_keys)

    def __iter__(self):
        return self._valid_keys


def parse_composite_model_expression(model_expression):
    """Parse the given model expression into a suitable model tree.

    Args:
        model_expression (str): the model expression string. Example:

        .. code-block:: none

            S0 * ( (Weight(Wball) * Ball) +
                   (Weight(Wstick) * Stick ) )

        If the model name is followed by parenthesis the string in parenthesis will represent the model's nickname.

    Returns:
        :class:`list`: the compartment model tree for use in composite models.
    """
    class Semantics:

        def expr(self, ast):
            if not isinstance(ast, list):
                return ast
            if isinstance(ast, list):
                return ast[0], ast[2], ast[1]
            return ast

        def term(self, ast):
            if not isinstance(ast, list):
                return ast
            if isinstance(ast, list):
                return ast[0], ast[2], ast[1]
            return ast

        def factor(self, ast):
            if isinstance(ast, list):
                return ast[1]
            return ast

        def model(self, ast):
            if isinstance(ast, str):
                return get_component('compartment_models', ast)()
            else:
                return get_component('compartment_models', ast[0])(ast[2])

    return _composite_model_expression_parser.parse(model_expression, semantics=Semantics())
