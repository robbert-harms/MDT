from copy import deepcopy
from mdt.component_templates.base import ComponentBuilder, ComponentTemplate
from mot.lib.cl_data_type import SimpleCLDataType
from mot.lib.cl_function import SimpleCLFunction
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
            template (LibraryFunctionTemplate): the library config template to use for creating
                the class with the right init settings.
        """
        class AutoCreatedLibraryFunction(CLLibrary, SimpleCLFunction):

            def __init__(self, *args, **kwargs):
                cl_code = template.cl_code or ''
                cl_extra = template.cl_extra or ''
                if not template.is_function:
                    cl_code = ''
                    cl_extra += '\n' + template.cl_code

                new_args = [template.return_type,
                            template.name,
                            _resolve_parameters(template.parameters),
                            cl_code,
                            ]

                for ind, already_set_arg in enumerate(args):
                    new_args[ind] = already_set_arg

                new_kwargs = dict(dependencies=_resolve_dependencies(template.dependencies),
                                  cl_extra=cl_extra)
                new_kwargs.update(kwargs)

                super().__init__(*new_args, **new_kwargs)

                if hasattr(template, 'init'):
                    template.init(self)

        for name, method in template.bound_methods.items():
            setattr(AutoCreatedLibraryFunction, name, method)

        return AutoCreatedLibraryFunction


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
        parameters (list): the list of parameters to use. If a parameter is a string we will
            use it automatically, if not it is supposed to be a LibraryParameter
            instance that we append directly.
        cl_code (str): the CL code definition to use.
        cl_extra (str): auxiliary functions for the library, prepended to the generated CL function.
        dependencies (list): the list of functions this function depends on, can contain string which will be
            resolved as library functions.
        is_function (boolean): set to False to disable the automatic generation of a function signature.
            Use this for C macro only libraries.
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
                parameters.append(LibraryParameter(SimpleCLDataType.from_string('mot_float_type'), item))
        else:
            parameters.append(deepcopy(item))
    return parameters
