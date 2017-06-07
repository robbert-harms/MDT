import inspect
import os
from copy import deepcopy

import six

from mdt.components_loader import ParametersLoader, ComponentConfigMeta, ComponentConfig, ComponentBuilder, \
    method_binding_meta
from mdt.models.compartments import DMRICompartmentModelFunction
from mot.model_building.model_function_priors import ModelFunctionPrior, SimpleModelFunctionPrior
from mot.model_building.parameters import CurrentObservationParam

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def _get_parameters_list(parameter_list):
    """Convert all the parameters in the given parameter list to actual parameter objects.

    Args:
        parameter_list (list): a list containing a mix of either parameter objects or strings. If it is a parameter
            we add a copy of it to the return list. If it is a string we will autoload it.

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
                parameters.append(parameters_loader.load(item))
        else:
            parameters.append(deepcopy(item))
    return parameters


def _construct_cl_function_definition(return_type, cl_function_name, parameters):
    """Create the CL function definition for a compartment function.

    This will construct something like (for the Stick model):

    .. code-block:: c

        double cmStick(const mot_float_type4 g,
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

    parameters_str = ',\n'.join(parameter_str(parameter) for parameter in parameters)
    return '{return_type} {cl_function_name}({parameters})'.format(return_type=return_type,
                                                                   cl_function_name=cl_function_name,
                                                                   parameters=parameters_str)


class CompartmentConfigMeta(ComponentConfigMeta):

    def __new__(mcs, name, bases, attributes):
        """Extends the default meta class with extra functionality for the compartments.

        This adds the cl_function_name if it is not defined, and creates the correct cl_code.
        """
        result = super(CompartmentConfigMeta, mcs).__new__(mcs, name, bases, attributes)

        if 'cl_function_name' not in attributes:
            result.cl_function_name = 'cm{}'.format(name)

        # to prevent the base from loading the initial meta class.
        if any(isinstance(base, CompartmentConfigMeta) for base in bases):
            result.cl_code = mcs._get_cl_code(result, bases, attributes)

        return result

    @classmethod
    def _get_cl_code(mcs, result, bases, attributes):
        return_type = CompartmentConfigMeta._resolve_attribute(bases, attributes, 'return_type') or 'double'

        if 'cl_code' in attributes and attributes['cl_code'] is not None:
            s = _construct_cl_function_definition(
                return_type, result.cl_function_name, _get_parameters_list(result.parameter_list))
            s += '{\n' + attributes['cl_code'] + '\n}'
            return s

        module_path = os.path.abspath(inspect.getfile(result))
        path = os.path.join(os.path.dirname(module_path), os.path.splitext(os.path.basename(module_path))[0]) + '.cl'
        if os.path.isfile(path):
            with open(path, 'r') as f:
                return f.read()

        for base in bases:
            if hasattr(base, 'cl_code') and base.cl_code is not None:
                return base.cl_code

    @staticmethod
    def _resolve_attribute(bases, attributes, attribute_name):
        if attribute_name in attributes:
            return attributes[attribute_name]
        for base in bases:
            if hasattr(base, attribute_name):
                return getattr(base, attribute_name)
        raise ValueError('Attribute not found in this component config or its superclasses.')


class CompartmentConfig(six.with_metaclass(CompartmentConfigMeta, ComponentConfig)):
    """The compartment config to inherit from.

    These configs are loaded on the fly by the CompartmentBuilder.

    All methods you define are automatically bound to the DMRICompartmentModelFunction. Also, to do extra
    initialization you can define a method ``init``. This method is called after object construction to allow
    for additional initialization and is not added to the final object.

    Attributes:
        name (str): the name of the model, defaults to the class name
        description (str): model description
        cl_function_name (str): the name of the function in the CL kernel
        parameter_list (list): the list of parameters to use. If a parameter is a string we will
            use it automatically, if not it is supposed to be a CLFunctionParameter
            instance that we append directly.
        cl_code (CLCodeDefinition): the CL code definition to use. Defaults to CLCodeFromAdjacentFile.
        dependency_list (list): the list of functions this function depends on, can contain string which will be
            resolved as library functions.
        return_type (str): the return type of this compartment, defaults to double.
        prior (str or None): an extra MCMC sampling prior for this compartment. This is additional to the priors
            defined in the parameters. This should be an instance of :class:`~mdt.models.compartments.CompartmentPrior`
            or a string with a CL function body. If the latter, the :class:`~mdt.models.compartments.CompartmentPrior`
            is automatically constructed based on the content of the string (automatic parameter recognition).
        post_optimization_modifiers (list): a list of modification callbacks for use after optimization. Examples:

            .. code-block:: python

                post_optimization_modifiers = [('FS', lambda d: 1 - d['w_ball.w']),
                                               ('Ball.d', lambda d: d['Ball.d'] * 1e9),
                                               (['Power2', 'Power3'], lambda d: [d['foo']**2, d['foo']**3]),
                                           ...]

            The last entry in the above example shows that it is possible to include more than one
            modifier in one modifier expression.

            These modifiers are supposed to be called before the modifiers of the composite model.
    """
    name = ''
    description = ''
    cl_function_name = None
    parameter_list = []
    cl_code = None
    dependency_list = []
    return_type = 'double'
    prior = None
    post_optimization_modifiers = None


class CompartmentBuildingBase(DMRICompartmentModelFunction):
    """Use this class in super calls if you want to overwrite methods in the inherited compartment configs.

    In python2 super needs a type to be able to do its work. This is the type you can give it to allow
    it to do its work.
    """


class CompartmentBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class CompartmentBuildingBase

        Args:
            template (CompartmentConfig): the compartment config template to use for
                creating the class with the right init settings.
        """
        class AutoCreatedDMRICompartmentModel(method_binding_meta(template, CompartmentBuildingBase)):

            def __init__(self, *args, **kwargs):
                parameter_list = _get_parameters_list(template.parameter_list)

                new_args = [template.name,
                            template.cl_function_name,
                            parameter_list,
                            template.cl_code,
                            _resolve_dependencies(template.dependency_list),
                            template.return_type]

                for ind, already_set_arg in enumerate(args):
                    new_args[ind] = already_set_arg

                new_kwargs = {'model_function_priors': (_resolve_prior(template.prior, template.name,
                                                                       [p.name for p in parameter_list],)),
                              'post_optimization_modifiers': template.post_optimization_modifiers}
                new_kwargs.update(kwargs)

                super(AutoCreatedDMRICompartmentModel, self).__init__(*new_args, **new_kwargs)

                if hasattr(template, 'init'):
                    template.init(self)

        return AutoCreatedDMRICompartmentModel


def _resolve_dependencies(dependency_list):
    """Resolve the dependency list such that the result contains all functions.

    Args:
        dependency_list (list): the list of dependencies as given by the user. Elements can either include actual
            instances of :class:`~mot.library_functions.CLLibrary` or strings with the name of the
            component to auto-load.

    Returns:
        list: a new list with the string elements resolved as :class:`~mot.library_functions.CLLibrary`.
    """
    from mdt.components_loader import LibraryFunctionsLoader

    lib_loader = LibraryFunctionsLoader()
    result = []
    for dependency in dependency_list:
        if isinstance(dependency, six.string_types):
            result.append(lib_loader.load(dependency))
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
