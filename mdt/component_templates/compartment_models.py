from copy import deepcopy, copy
from textwrap import indent, dedent
import numpy as np
import six

from mdt.components_loader import ParametersLoader
from mdt.component_templates.base import ComponentBuilder, method_binding_meta, ComponentTemplate, register_builder
from mdt.models.compartments import DMRICompartmentModelFunction
from mdt.utils import spherical_to_cartesian
from mot.model_building.model_function_priors import ModelFunctionPrior, SimpleModelFunctionPrior
from mot.model_building.parameters import CurrentObservationParam

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CompartmentTemplate(ComponentTemplate):
    """The compartment config to inherit from.

    These configs are loaded on the fly by the CompartmentBuilder.

    All methods you define are automatically bound to the DMRICompartmentModelFunction. Also, to do extra
    initialization you can define a method ``init``. This method is called after object construction to allow
    for additional initialization and is not added to the final object.

    Attributes:
        name (str): the name of the model, defaults to the class name
        description (str): model description
        parameter_list (list): the list of parameters to use. If a parameter is a string we will
            use it automatically, if not it is supposed to be a CLFunctionParameter
            instance that we append directly.
        cl_code (str): the CL code definition to use, please provide here the body of your CL function.
        cl_extra (str): additional CL code for your model. This will be prepended to the body of your CL function.
        dependency_list (list): the list of functions this function depends on, can contain string which will be
            resolved as library functions.
        return_type (str): the return type of this compartment, defaults to double.
        extra_prior (str or None): an extra MCMC sampling prior for this compartment. This is additional to the priors
            defined in the parameters. This should be an instance of :class:`~mdt.models.compartments.CompartmentPrior`
            or a string with a CL function body. If the latter, the :class:`~mdt.models.compartments.CompartmentPrior`
            is automatically constructed based on the content of the string (automatic parameter recognition).
        post_optimization_modifiers (list): a list of modification callbacks for use after optimization. Examples:

            .. code-block:: python

                post_optimization_modifiers = [('FS', lambda d: 1 - d['w_ball.w']),
                                               ('Kurtosis.MK', lambda d, protocol: <...>),
                                               (['Power2', 'Power3'], lambda d: [d['foo']**2, d['foo']**3]),
                                           ...]

            The last entry in the above example shows that it is possible to include more than one
            modifier in one modifier expression. In general, the function given should accept as first argument
            the results dictionary and as optional second argument the protocol used to generate the results.
            These modifiers are called before the modifiers of the composite model.

        auto_add_cartesian_vector (boolean): if set to True we will automatically add a post optimization modifier
            that constructs a cartesian vector from the ``theta`` and ``phi`` parameter if present. This modifier
            is run before the other user defined modifiers.

        sampling_covar_extras (list): list with information about callback functions that can add additional maps
            to the covariance matrix calculated after sampling. Usage example::

                sampling_covar_extras = [(('theta', 'phi'),
                                          ('vec0_x', 'vec0_y', 'vec0_z'),
                                          spherical_to_cartesian),
                                         ...]

            This requires a list of tuples with in those tuples three elements: the names of the parameters
            to use as input to the callback function, the names of the output parameters and finally the callback
             function itself.

        sampling_covar_exclude (None tuple or list): parameters to exclude in the covariance matrix calculation
                after sampling. Example::

                sampling_covar_exclude = ['theta', 'phi']

        auto_sampling_covar_cartesian (boolean): if set to True we automatically use cartesian coordinates for
            the sampling covariance matrix instead of the spherical coordinates.
    """
    name = ''
    description = ''
    parameter_list = []
    cl_code = None
    cl_extra = None
    dependency_list = []
    return_type = 'double'
    extra_prior = None
    post_optimization_modifiers = None
    auto_add_cartesian_vector = True
    sampling_covar_extras = None
    sampling_covar_exclude = None
    auto_sampling_covar_cartesian = True


class CompartmentBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class CompartmentBuildingBase

        Args:
            template (CompartmentTemplate): the compartment config template to use for
                creating the class with the right init settings.
        """
        builder = self

        class AutoCreatedDMRICompartmentModel(method_binding_meta(template, DMRICompartmentModelFunction)):

            def __init__(self, *args, **kwargs):
                parameter_list = _get_parameters_list(template.parameter_list)

                new_args = [template.name,
                            template.name,
                            parameter_list,
                            _build_source_code(template),
                            _resolve_dependencies(template.dependency_list),
                            template.return_type]

                for ind, already_set_arg in enumerate(args):
                    new_args[ind] = already_set_arg

                covar_extras, covar_exclude = _resolve_covariance_extra_exclude(template, parameter_list)

                new_kwargs = {'model_function_priors': (_resolve_prior(template.extra_prior, template.name,
                                                                       [p.name for p in parameter_list],)),
                              'post_optimization_modifiers': builder._get_post_optimization_modifiers(template,
                                                                                                      parameter_list),
                              'sampling_covar_extras': covar_extras,
                              'sampling_covar_exclude': covar_exclude}
                new_kwargs.update(kwargs)

                super(AutoCreatedDMRICompartmentModel, self).__init__(*new_args, **new_kwargs)

                if hasattr(template, 'init'):
                    template.init(self)

        return AutoCreatedDMRICompartmentModel

    def _get_post_optimization_modifiers(self, template, parameter_list):
        post_optimization_modifiers = []
        if getattr(template, 'auto_add_cartesian_vector', False):
            if all(map(lambda name: name in [p.name for p in parameter_list], ('theta', 'phi'))):
                modifier = ('vec0', lambda results: spherical_to_cartesian(np.squeeze(results['theta']),
                                                                           np.squeeze(results['phi'])))
                post_optimization_modifiers.append(modifier)

        if template.post_optimization_modifiers:
            post_optimization_modifiers.extend(template.post_optimization_modifiers)
        return post_optimization_modifiers


register_builder(CompartmentTemplate, CompartmentBuilder())


def _resolve_dependencies(dependency_list):
    """Resolve the dependency list such that the result contains all functions.

    Args:
        dependency_list (list): the list of dependencies as given by the user. Elements can either include actual
            instances of :class:`~mot.library_functions.CLLibrary` or strings with the name of libraries or
            other compartments to load.

    Returns:
        list: a new list with the string elements resolved as :class:`~mot.library_functions.CLLibrary`.
    """
    from mdt.components_loader import LibraryFunctionsLoader, CompartmentModelsLoader

    lib_loader = LibraryFunctionsLoader()
    compartment_loader = CompartmentModelsLoader()

    result = []
    for dependency in dependency_list:
        if isinstance(dependency, six.string_types):
            if lib_loader.has_component(dependency):
                result.append(lib_loader.load(dependency))
            else:
                result.append(compartment_loader.load(dependency))
        else:
            result.append(dependency)

    return result


def _resolve_prior(prior, compartment_name, compartment_parameters):
    """Create a proper prior out of the given prior information.

    Args:
        prior (str or mdt.models.compartments.CompartmentPrior or None):
            The prior from which to construct a prior.
        compartment_name (str): the name of the compartment
        compartment_parameters (list of str): the list of parameters of this compartment, used
            for looking up the used parameters in a string prior

    Returns:
        None or mdt.models.compartments.CompartmentPrior: a proper prior instance
    """
    if prior is None:
        return None

    if isinstance(prior, ModelFunctionPrior):
        return prior

    parameters = [p for p in compartment_parameters if p in prior]
    return SimpleModelFunctionPrior(prior, parameters, 'prior_' + compartment_name)


def _resolve_covariance_extra_exclude(template, parameter_list):
    """Resolves the defined covariance extra and exclude definitions.

    If ``auto_sampling_covar_cartesian`` is defined this function sets the ``sampling_covar_extras``
    and ``sampling_covar_exclude``

    Args:
        template (CompartmentTemplate): the template to use
        parameter_list (list): the list of parameters in the model

    Returns:
        tuple: sane values for ``sampling_covar_extras`` and ``sampling_covar_exclude``
    """
    extras = copy(template.sampling_covar_extras) or []
    excludes = copy(template.sampling_covar_exclude) or []

    def conversion_func(theta, phi):
        return np.rollaxis(spherical_to_cartesian(theta, phi), 2, 1)

    if template.auto_sampling_covar_cartesian:
        if all(map(lambda name: name in [p.name for p in parameter_list], ('theta', 'phi'))):
            excludes.extend(['theta', 'phi'])
            extras.append((('theta', 'phi'), ('vec0_x', 'vec0_y', 'vec0_z'), conversion_func))

    return extras, excludes


def _build_source_code(template):
    """Build the full model source code for the given compartment template.

    Args:
        template (CompartmentTemplate): the template for which to construct the CL code

    Returns:
        str: the model code
    """
    s = ''
    if template.cl_extra:
        s += template.cl_extra
    s += _construct_cl_function_definition(template.return_type, template.name,
                                          _get_parameters_list(template.parameter_list))
    s += '{\n\n' + indent(dedent(template.cl_code.strip('\n')), ' ' * 4) + '\n}'
    return s


def _get_parameters_list(parameter_list):
    """Convert all the parameters in the given parameter list to actual parameter objects.

    Args:
        parameter_list (list): a list containing a mix of either parameter objects, strings or tuples. If it is a
            parameter we add a copy of it to the return list. If it is a string we will autoload it. It is possible to
            specify a nickname for that parameter in this compartment using the syntax: ``<param>(<nickname>)``.

    Returns:
        list: the list of actual parameter objects
    """
    parameters_loader = ParametersLoader()

    parameters = []
    for item in parameter_list:
        if isinstance(item, six.string_types):
            if item == '_observation':
                parameters.append(CurrentObservationParam())
            else:
                if '(' in item:
                    param_name = item[:item.index('(')].strip()
                    nickname = item[item.index('(')+1:item.index(')')].strip()
                else:
                    param_name = item
                    nickname = None
                parameters.append(parameters_loader.load(param_name, nickname=nickname))
        else:
            parameters.append(deepcopy(item))
    return parameters


def _construct_cl_function_definition(return_type, cl_function_name, parameters):
    """Create the CL function definition for a compartment function.

    This will construct something like (for the Stick model):

    .. code-block:: c

        double Stick(const mot_float_type4 g,
                     const mot_float_type b,
                     const mot_float_type d,
                     const mot_float_type theta,
                     const mot_float_type phi)

    Args:
        return_type (str): the return type
        cl_function_name (str): the name of the function
        parameters (list of CLFunctionParameter): the list of function parameters we use for the arguments

    Returns:
        str: the function definition (only the signature).
    """
    def parameter_str(parameter):
        s = parameter.data_type.cl_type

        if parameter.data_type.pre_data_type_type_qualifiers:
            for qualifier in parameter.data_type.pre_data_type_type_qualifiers:
                s = qualifier + ' ' + s

        if parameter.data_type.address_space_qualifier:
            s = parameter.data_type.address_space_qualifier + ' ' + s

        if parameter.data_type.post_data_type_type_qualifier:
            s += ' ' + parameter.data_type.post_data_type_type_qualifier

        s += ' ' + parameter.name

        return s

    parameters_str = indent(',\n'.join(parameter_str(parameter) for parameter in parameters), ' '*4*2)
    return '\n{return_type} {cl_function_name}(\n{parameters})'.format(
        return_type=return_type, cl_function_name=cl_function_name, parameters=parameters_str)

