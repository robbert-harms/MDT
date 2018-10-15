from copy import deepcopy
from mdt.component_templates.base import ComponentBuilder, ComponentTemplate
from mot.lib.cl_data_type import SimpleCLDataType
from mot.lib.cl_function import SimpleCLFunction, SimpleCLFunctionParameter, SimpleCLCodeObject
from mdt.model_building.parameters import LibraryParameter
from mdt.lib.components import get_component, has_component
from mot.library_functions.base import CLLibrary

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LibraryFunctionsBuilder(ComponentBuilder):

    def _create_class(self, template):
        """Creates classes with as base class LibraryFunctionsBase

        Args:
            template (LibraryFunctionTemplate): the library config template to use for creating the library function
        """
        if template.is_function:
            class AutoCreatedLibraryFunction(CLLibrary, SimpleCLFunction):
                def __init__(self):
                    dependencies = _resolve_dependencies(template.dependencies)

                    if template.cl_extra:
                        extra_code = '''
                            #ifndef {inclusion_guard_name}
                            #define {inclusion_guard_name}
                            {cl_extra}
                            #endif // {inclusion_guard_name}
                        '''.format(inclusion_guard_name='INCLUDE_GUARD_{}_EXTRA'.format(template.name),
                                   cl_extra=template.cl_extra)
                        dependencies.append(SimpleCLCodeObject(extra_code))

                    super().__init__(
                        template.return_type, template.name,
                        _resolve_parameters(template.parameters), template.cl_code,
                        dependencies=dependencies)
        else:
            class AutoCreatedLibraryFunction(SimpleCLCodeObject):
                def __init__(self):
                    str = ''
                    if template.cl_extra is not None:
                        str += template.cl_extra
                    if template.cl_code is not None:
                        str += template.cl_code

                    cl_code = '''
                        #ifndef {inclusion_guard_name}
                        #define {inclusion_guard_name}
                        {cl_code}
                        #endif // {inclusion_guard_name}
                    '''.format(inclusion_guard_name='INCLUDE_GUARD_{}'.format(template.name),
                               cl_code=str)

                    super().__init__(cl_code)

        for name, method in template.bound_methods.items():
            setattr(AutoCreatedLibraryFunction, name, method)

        return AutoCreatedLibraryFunction


class LibraryFunctionTemplate(ComponentTemplate):
    """The library function config to inherit from.

    These configs are loaded on the fly by the LibraryFunctionsBuilder.

    Attributes:
        name (str): the name of the model, defaults to the class name
        description (str): model description
        return_type (str): the return type of the function, defaults to ``void``
        parameters (list): the list of parameters to use. If a parameter is a string we will
            use it automatically, if not it is supposed to be a LibraryParameter
            instance that we append directly.
        cl_code (str): the CL code definition to use.
        cl_extra (str): auxiliary functions for the library, prepended to the generated CL function.
        dependencies (list): the list of functions this function depends on, can contain string which will be
            resolved as library functions.
        is_function (boolean): set to False to disable the automatic generation of a function signature.
            Use this for macro or typedef only libraries.
    """
    _component_type = 'library_functions'
    _builder = LibraryFunctionsBuilder()

    name = ''
    description = ''
    return_type = 'void'
    parameters = []
    cl_code = None
    cl_extra = None
    dependencies = []
    is_function = True


def _resolve_dependencies(dependencies):
    """Resolve the dependency list such that the result contains all functions.

    Args:
        dependencies (list): the list of dependencies as given by the user. Elements can either include actual
            instances of :class:`~mot.library_functions.CLLibrary` or strings with the name of the
            component to auto-load.

    Returns:
        list: a new list with the string elements resolved
            as :class:`~mot.library_functions.CLLibrary`.
    """
    result = []
    for dependency in dependencies:
        if isinstance(dependency, str):
            result.append(get_component('library_functions', dependency)())
        else:
            result.append(dependency)

    return result


def _resolve_parameters(parameter_list):
    """Convert all the parameters in the given parameter list to actual parameter objects.

    This will load all the parameters as :class:`~mdt.model_building.parameters.LibraryParameter`.

    Args:
        parameter_list (list): a list containing a mix of either parameter objects or strings. If it is a parameter
            we add a copy of it to the return list. If it is a string we will autoload it.

    Returns:
        list: the list of actual parameter objects
    """
    parameters = []
    for item in parameter_list:
        if isinstance(item, str):
            if has_component('parameters', item):
                param = get_component('parameters', item)()
                parameters.append(LibraryParameter(param.data_type, item))
            else:
                parameters.append(LibraryParameter(SimpleCLDataType.from_string('double'), item))
        elif isinstance(item, (tuple, list)):
            parameters.append(SimpleCLFunctionParameter(item[0], item[1]))
        else:
            parameters.append(deepcopy(item))
    return parameters
