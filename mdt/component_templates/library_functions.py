from copy import deepcopy
from textwrap import indent, dedent

import six
from mdt.component_templates.base import ComponentBuilder, method_binding_meta, \
    ComponentTemplate, register_builder
from mdt.components_loader import ParametersLoader
from mot.cl_data_type import SimpleCLDataType
from mot.library_functions import SimpleCLLibrary
from mot.model_building.parameters import LibraryParameter

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LibraryFunctionTemplate(ComponentTemplate):
    """The library function config to inherit from.

    These configs are loaded on the fly by the LibraryFunctionsBuilder.

    All methods you define are automatically bound to the SimpleCLLibrary. Also, to do extra
    initialization you can define a method ``init``. This method is called after object construction to allow
    for additional initialization and is is not added to the final object.

    Attributes:
        name (str): the name of the model, defaults to the class name
        description (str): model description
        return_type (str): the return type of the function, defaults to ``void``
        parameter_list (list): the list of parameters to use. If a parameter is a string we will
            use it automatically, if not it is supposed to be a LibraryParameter
            instance that we append directly.
        cl_code (str): the CL code definition to use.
        cl_extra (str): auxiliary functions for the library, prepended to the generated CL function.
        dependency_list (list): the list of functions this function depends on, can contain string which will be
            resolved as library functions.
        is_function (boolean): set to False to disable the automatic generation of a function signature.
            Use this for C macro only libraries.
    """
    name = ''
    description = ''
    return_type = 'void'
    parameter_list = []
    cl_code = None
    cl_extra = None
    dependency_list = []
    is_function = True


class LibraryFunctionBuildingBase(SimpleCLLibrary):
    """Use this class in super calls if you want to overwrite methods in the inherited compartment configs.

    In python2 super needs a type to be able to do its work. This is the type you can give it to allow
    it to do its work.
    """


class LibraryFunctionsBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class LibraryFunctionsBase

        Args:
            template (LibraryFunctionTemplate): the library config template to use for creating
                the class with the right init settings.
        """
        class AutoCreatedLibraryFunction(method_binding_meta(template, LibraryFunctionBuildingBase)):

            def __init__(self, *args, **kwargs):
                new_args = [template.name,
                            _build_source_code(template),
                            ]

                for ind, already_set_arg in enumerate(args):
                    new_args[ind] = already_set_arg

                new_kwargs = dict(dependencies=_resolve_dependencies(template.dependency_list))
                new_kwargs.update(kwargs)

                super(AutoCreatedLibraryFunction, self).__init__(*new_args, **new_kwargs)

                if hasattr(template, 'init'):
                    template.init(self)

        return AutoCreatedLibraryFunction


register_builder(LibraryFunctionTemplate, LibraryFunctionsBuilder())


def _resolve_dependencies(dependency_list):
    """Resolve the dependency list such that the result contains all functions.

    Args:
        dependency_list (list): the list of dependencies as given by the user. Elements can either include actual
            instances of :class:`~mot.library_functions.CLLibrary` or strings with the name of the
            component to auto-load.

    Returns:
        list: a new list with the string elements resolved
            as :class:`~mot.library_functions.CLLibrary`.
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


def _get_parameters_list(parameter_list):
    """Convert all the parameters in the given parameter list to actual parameter objects.

    This will load all the parameters as :class:`~mot.model_building.parameters.LibraryParameter`.

    Args:
        parameter_list (list): a list containing a mix of either parameter objects or strings. If it is a parameter
            we add a copy of it to the return list. If it is a string we will autoload it.

    Returns:
        list: the list of actual parameter objects
    """
    param_loader = ParametersLoader()

    parameters = []
    for item in parameter_list:
        if isinstance(item, six.string_types):
            if param_loader.has_component(item):
                param = param_loader.load(item)
                parameters.append(LibraryParameter(param.data_type, item))
            else:
                parameters.append(LibraryParameter(SimpleCLDataType.from_string('mot_float_type'), item))
        else:
            parameters.append(deepcopy(item))
    return parameters


def _construct_cl_function_definition(return_type, cl_function_name, parameters):
    """Create the CL function definition for a compartment function.

    This will construct something like (for the NeumannCylindricalRestrictedSignal model):

    .. code-block:: c

        double NeumannCylindricalRestrictedSignal(
                const mot_float_type Delta,
                const mot_float_type delta,
                const mot_float_type d,
                const mot_float_type R)

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

    parameters_str = indent(',\n'.join(parameter_str(parameter) for parameter in parameters), ' ' * 4 * 2)
    return '\n{return_type} {cl_function_name}(\n{parameters})'.format(
        return_type=return_type, cl_function_name=cl_function_name, parameters=parameters_str)


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
    if template.is_function:
        s += _construct_cl_function_definition(template.return_type, template.name,
                                               _get_parameters_list(template.parameter_list))
        s += '{\n\n' + indent(dedent(template.cl_code.strip('\n')), ' ' * 4) + '\n}'
    else:
        s += '\n' + dedent(template.cl_code.strip('\n'))
    return s
