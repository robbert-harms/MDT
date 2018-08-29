from copy import deepcopy, copy
import numpy as np
from mdt.component_templates.base import ComponentBuilder, ComponentTemplate
from mdt.lib.components import get_component, has_component
from mdt.models.compartments import DMRICompartmentModelFunction
from mdt.utils import spherical_to_cartesian
from mot.lib.cl_function import CLFunction, SimpleCLFunction
from mdt.model_building.model_functions import WeightType
from mdt.model_building.parameters import CurrentObservationParam

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CompartmentBuilder(ComponentBuilder):

    def _create_class(self, template):
        """Creates classes with as base class CompartmentBuildingBase

        Args:
            template (CompartmentTemplate): the compartment config template to use for
                creating the class with the right init settings.
        """
        builder = self

        class AutoCreatedDMRICompartmentModel(DMRICompartmentModelFunction):

            def __init__(self, *args, **kwargs):
                parameters = []
                if len(template.parameters):
                    parameters = _resolve_parameters(template.parameters)

                dependencies = []
                if len(template.dependencies):
                    dependencies = _resolve_dependencies(template.dependencies)

                new_args = [template.name,
                            template.name,
                            parameters,
                            template.cl_code,
                            dependencies,
                            template.return_type]

                for ind, already_set_arg in enumerate(args):
                    new_args[ind] = already_set_arg

                new_kwargs = {
                    'model_function_priors': (_resolve_prior(template.extra_prior, template.name,
                                                             [p.name for p in parameters],)),
                    'post_optimization_modifiers': template.post_optimization_modifiers,
                    'extra_optimization_maps_funcs': builder._get_extra_optimization_map_funcs(
                        template, parameters),
                    'extra_sampling_maps_funcs': copy(template.extra_sampling_maps),
                    'cl_extra': template.cl_extra,
                    'proposal_callbacks': builder._get_proposal_callbacks(template, parameters)}
                new_kwargs.update(kwargs)

                super().__init__(*new_args, **new_kwargs)

                if hasattr(template, 'init'):
                    template.init(self)

        for name, method in template.bound_methods.items():
            setattr(AutoCreatedDMRICompartmentModel, name, method)

        return AutoCreatedDMRICompartmentModel

    def _get_extra_optimization_map_funcs(self, template, parameter_list):
        extra_optimization_maps = copy(template.extra_optimization_maps)
        if all(map(lambda name: name in [p.name for p in parameter_list], ('theta', 'phi'))):
            extra_optimization_maps.append(lambda results: {
                'vec0': spherical_to_cartesian(np.squeeze(results['theta']), np.squeeze(results['phi']))})
        return extra_optimization_maps

    def _get_proposal_callbacks(self, template, parameter_list):
        """Get a list of proposal callback functions.

        These functions are (indirectly) called by a MCMC sample routine to finalize the new proposal vector.

        Returns:
            List[(Tuple, mot.lib.cl_function.CLFunction)]: a list of proposal callback functions coupled with
                references to the compartment parameters used in the function.
        """
        callbacks = []

        def existing_parameters(param_list):
            param_names = [p.name for p in parameter_list]
            return all(p in param_names for p in param_list)

        def get_corresponding_param(param_name):
            for p in parameter_list:
                if p.name == param_name:
                    return p

        if template.spherical_parameters is not None and existing_parameters(template.spherical_parameters):
            corresponding_params = [get_corresponding_param(p) for p in template.spherical_parameters]

            func = SimpleCLFunction(
                'void',
                'proposal_callback_spherical_{}'.format(template.name),
                [('mot_float_type*', 'theta'), ('mot_float_type*', 'phi')],
                '''
                    mot_float_type oldtheta = *theta;
                    mot_float_type oldphi = *phi;
                
                    if(*phi > M_PI_F){
                        *phi -= M_PI_F;
                        *theta = M_PI_F - *theta;
                    }
                    else if(*phi < 0){
                        *phi += M_PI_F;
                        *theta = M_PI_F - *theta;
                    }
            ''')
            callbacks.append((corresponding_params, func))

        for p in parameter_list:
            if hasattr(p, 'sampling_proposal_modulus') and p.sampling_proposal_modulus is not None:
                func = SimpleCLFunction(
                    'void',
                    'proposal_callback_{}_{}'.format(template.name, p.name),
                    [('mot_float_type*', p.name)],
                    '*{0} = *{0} - floor(*{0} / {1}) * {1};'.format(p.name, p.sampling_proposal_modulus))

                callbacks.append(([p], func))

        return callbacks


class WeightBuilder(ComponentBuilder):
    def _create_class(self, template):

        class AutoCreatedWeightModel(WeightType):

            def __init__(self, *args, **kwargs):
                parameters = []
                if len(template.parameters):
                    parameters = _resolve_parameters(template.parameters)

                dependencies = []
                if len(template.dependencies):
                    dependencies = _resolve_dependencies(template.dependencies)

                new_args = [template.name,
                            template.name,
                            parameters,
                            template.cl_code,
                            ]

                for ind, already_set_arg in enumerate(args):
                    new_args[ind] = already_set_arg

                new_kwargs = {
                    'dependencies': dependencies,
                    'cl_extra': template.cl_extra}
                new_kwargs.update(kwargs)

                super().__init__(template.return_type, *new_args, **new_kwargs)

                if hasattr(template, 'init'):
                    template.init(self)

        for name, method in template.bound_methods.items():
            setattr(AutoCreatedWeightModel, name, method)

        return AutoCreatedWeightModel


class CompartmentTemplate(ComponentTemplate):
    """The compartment config to inherit from.

    These configs are loaded on the fly by the CompartmentBuilder.

    All methods you define are automatically bound to the DMRICompartmentModelFunction. Also, to do extra
    initialization you can define a method ``init``. This method is called after object construction to allow
    for additional initialization and is not added to the final object.

    Attributes:
        name (str): the name of the model, defaults to the class name

        description (str): model description

        parameters (list): the list of parameters to use. If a parameter is a string we will
            use it automatically, if not it is supposed to be a CLFunctionParameter
            instance that we append directly.

        cl_code (str): the CL code definition to use, please provide here the body of your CL function.

        cl_extra (str): additional CL code for your model. This will be prepended to the body of your CL function.

        dependencies (list): the list of functions this function depends on, can contain string which will be
            resolved as library functions.

        return_type (str): the return type of this compartment, defaults to double.

        extra_prior (str or None): an extra MCMC sample prior for this compartment. This is additional to the priors
            defined in the parameters. This should be an instance of :class:`~mdt.models.compartments.CompartmentPrior`
            or a string with a CL function body. If the latter, the :class:`~mdt.models.compartments.CompartmentPrior`
            is automatically constructed based on the content of the string (automatic parameter recognition).

        post_optimization_modifiers (list): a list of modification callbacks to change the estimated
            optimization points. This should not add new maps, for that use
            the ``extra_optimization_maps_funcs`` directive. This directive can be used to, for example,
            sort maps into a similar valued but sorted representation (actually the ``sort_maps`` directive
            creates a post_optimization_modifier)

            Examples:

            .. code-block:: python

                post_optimization_modifiers = [lambda d: reorient_tensor(d),
                                               lambda d: sort_maps(d)
                                           ...]

            Each function should accept the results dictionary as input and return a dictionary with new maps.

        extra_optimization_maps (list): a list of functions to return extra information maps based on a point estimate.
            This is called after the post optimization modifiers and after the model calculated uncertainties based
            on the Fisher Information Matrix. Therefore, these routines can propagate uncertainties in the estimates.
            Please note that this is only for adding additional maps. For changing the point estimate of the
            optimization, please use the ``post_optimization_modifiers`` directive.

            Examples:

            .. code-block:: python

                extra_optimization_maps_funcs = [lambda d: {'FS': 1 - d['w']},
                                                 lambda d: {'Kurtosis.MK': <...>},
                                                 lambda d: {'Power2': d['foo']**2, 'Power3': d['foo']**3},
                                                 ...]

        extra_sampling_maps (list): a list of functions to return additional maps as results from sample.
            This is called after sample with as argument a dictionary containing the sample results and
            the values of the fixed parameters.

            Examples::

                extra_sampling_maps = [lambda s: {'MD': np.mean((s['d'] + s['dperp0'] + s['dperp1'])/3., axis=1)}
                                      ...]

        spherical_parameters (tuple or None): if None, this feature is disabled. If set, it should be a tuple
            with two elements for the name of the inclination and the azimuth parameters in the compartment model
            (defaults to 'theta' and 'phi'). If set to such a tuple, the compartment model will automatically add a
            proposal callback function that ensures that the these spherical coordinates are sampled correctly for
            values near theta == PI. Additionally, this callback function is used when computing the numerical
            derivative.

    """
    _component_type = 'compartment_models'
    _builder = CompartmentBuilder()

    name = ''
    description = ''
    parameters = []
    cl_code = None
    cl_extra = None
    dependencies = []
    return_type = 'double'
    extra_prior = None
    post_optimization_modifiers = []
    extra_optimization_maps = []
    extra_sampling_maps = []
    spherical_parameters = ('theta', 'phi')


class WeightCompartmentTemplate(ComponentTemplate):
    """Special compartment template for representing a Weight.

    Defining a compartment as a Weight enables automatic volume fraction weighting, and ensures that all weights sum
    to one during optimization and sample.
    """
    _component_type = 'compartment_models'
    _builder = WeightBuilder()

    name = ''
    description = ''
    parameters = []
    cl_code = None
    cl_extra = None
    dependencies = []
    return_type = 'double'


def _resolve_dependencies(dependencies):
    """Resolve the dependency list such that the result contains all functions.

    Args:
        dependencies (list): the list of dependencies as given by the user. Elements can either include actual
            instances of :class:`~mot.library_functions.CLLibrary` or strings with the name of libraries or
            other compartments to load.

    Returns:
        list: a new list with the string elements resolved as :class:`~mot.library_functions.CLLibrary`.
    """
    result = []
    for dependency in dependencies:
        if isinstance(dependency, str):
            if has_component('library_functions', dependency):
                result.append(get_component('library_functions', dependency)())
            else:
                result.append(get_component('compartment_models', dependency)())
        else:
            result.append(dependency)

    return result


def _resolve_prior(prior, compartment_name, compartment_parameters):
    """Create a proper prior out of the given prior information.

    Args:
        prior (str or mot.lib.cl_function.CLFunction or None):
            The prior from which to construct a prior.
        compartment_name (str): the name of the compartment
        compartment_parameters (list of str): the list of parameters of this compartment, used
            for looking up the used parameters in a string prior

    Returns:
        None or mdt.models.compartments.CompartmentPrior: a proper prior instance
    """
    if prior is None:
        return None

    if isinstance(prior, CLFunction):
        return prior

    parameters = [('mot_float_type', p) for p in compartment_parameters if p in prior]
    return SimpleCLFunction('mot_float_type', 'prior_' + compartment_name, parameters, prior)


def _resolve_parameters(parameter_list):
    """Convert all the parameters in the given parameter list to actual parameter objects.

    Args:
        parameter_list (list): a list containing a mix of either parameter objects, strings or tuples. If it is a
            parameter we add a copy of it to the return list. If it is a string we will autoload it. It is possible to
            specify a nickname for that parameter in this compartment using the syntax: ``<param>(<nickname>)``.

    Returns:
        list: the list of actual parameter objects
    """
    parameters = []
    for item in parameter_list:
        if isinstance(item, str):
            if item == '_observation':
                parameters.append(CurrentObservationParam())
            else:
                if '(' in item:
                    param_name = item[:item.index('(')].strip()
                    nickname = item[item.index('(')+1:item.index(')')].strip()
                else:
                    param_name = item
                    nickname = None
                parameters.append(get_component('parameters', param_name)(nickname=nickname))
        else:
            parameters.append(deepcopy(item))
    return parameters
