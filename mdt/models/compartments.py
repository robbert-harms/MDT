import inspect

from pkg_resources import resource_filename
import os
from copy import deepcopy
import six

from mdt.components_loader import ComponentConfig, ComponentBuilder, ParametersLoader
from mdt.utils import spherical_to_cartesian
from mot.base import ModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-12-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICompartmentModelFunction(ModelFunction):

    def __init__(self, name, cl_function_name, parameter_list, cl_header, cl_code, dependency_list):
        """Create a new dMRI compartment model function.

        Args:
            name (str): the name of this compartment model
            cl_function_name (str): the name of this function in the CL kernel
            parameter_list (list of CLFunctionParameter): the list of the function parameters
            cl_header (str): the code for the CL header
            cl_code (str): the code for the function in CL
            dependency_list (list): the list of functions we depend on inside the kernel
        """
        super(DMRICompartmentModelFunction, self).__init__(name, cl_function_name, parameter_list,
                                                           dependency_list=dependency_list)
        self._cl_header = cl_header
        self._cl_code = cl_code

    def get_cl_header(self):
        inclusion_guard_name = 'DMRICM_' + self.cl_function_name + '_H'
        return '''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {header}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=self._get_cl_dependency_headers(), inclusion_guard_name=inclusion_guard_name,
                   header=self._cl_header)

    def get_cl_code(self):
        inclusion_guard_name = 'DMRICM_' + self.cl_function_name + '_CL'
        return '''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {header}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=self._get_cl_dependency_code(), inclusion_guard_name=inclusion_guard_name,
                   header=self._cl_code)

    def _get_vector_result_maps(self, theta, phi):
        """Convert spherical coordinates to cartesian vector in 3d

        Args:
            theta (ndarray): the double array with the theta values
            phi (ndarray): the double array with the phi values

        Returns:
            dict: containing the cartesian vector representing the fibre direction in multiple forms.
        """
        cartesian = spherical_to_cartesian(theta, phi)
        extra_dict = {self.name + '.vec0': cartesian}

        for ind in range(3):
            extra_dict.update({self.name + '.vec0_' + repr(ind): cartesian[:, ind]})

        return extra_dict


class CLCodeDefinition(object):

    def get_code(self, config):
        """Get the CL code for this compartment model.

        Args:
            config (CompartmentConfig): the compartment configuration

        Returns:
            str: the compartment model code.
        """


class CLCodeFromString(CLCodeDefinition):

    def __init__(self, cl_code_str):
        self.cl_code_str = cl_code_str

    def get_code(self, config):
        return self.cl_code_str


class CLCodeFromInlineString(CLCodeDefinition):

    def __init__(self, cl_inline_code_str):
        self.cl_inline_code_str = cl_inline_code_str

    def get_code(self, config):
        s = _construct_cl_function_definition('mot_float_type',
                                              config.cl_function_name,
                                              _get_parameters_list(config.parameter_list))
        s += '{\n' + self.cl_inline_code_str + '\n}'
        return s


class CLCodeFromFile(CLCodeDefinition):

    def __init__(self, cl_code_file):
        self.cl_code_file = cl_code_file

    def get_code(self, config):
        with open(os.path.abspath(self.cl_code_file), 'r') as f:
            return f.read()


class CLCodeFromAdjacentFile(CLCodeDefinition):

    def get_code(self, config):
        module_path = os.path.abspath(inspect.getfile(config))
        path = os.path.join(os.path.dirname(module_path), os.path.splitext(os.path.basename(module_path))[0])
        with open(path + '.cl', 'r') as f:
            return f.read()


class CLHeaderDefinition(object):

    def get_code(self, config):
        """Get the CL header code for this compartment model.

        Args:
            config (CompartmentConfig): the compartment configuration

        Returns:
            str: the compartment model code.
        """


class CLHeaderFromTemplate(CLCodeDefinition):

    def get_code(self, config):
        return _construct_cl_function_definition('mot_float_type',
                                                 config.cl_function_name,
                                                 _get_parameters_list(config.parameter_list)) + ';'


class CLHeaderFromFile(CLCodeDefinition):

    def __init__(self, cl_code_file):
        self.cl_code_file = cl_code_file

    def get_code(self, config):
        with open(os.path.abspath(self.cl_code_file), 'r') as f:
            return f.read()


class CLHeaderFromAdjacentFile(CLCodeDefinition):

    def __init__(self, module_name):
        self.module_name = module_name

    def get_code(self, config):
        with open(os.path.abspath(resource_filename(self.module_name, config.name + '.h')), 'r') as f:
            return f.read()


class CompartmentConfig(ComponentConfig):
    """The compartment config to inherit from.

    These configs are loaded on the fly by the CompartmentBuilder.

    All methods you define are automatically bound to the DMRICompartmentModelFunction. Also, to do extra
    initialization you can define a method init. This method is called after object construction to allow
    for additional initialization. Also, this method is not added to the final object.

    Class attributes:
        name (str): the name of the model
        description (str): model description
        cl_function_name (str): the name of the function in the CL kernel
        parameter_list (list): the list of parameters to use. If a parameter is a string we will load it automatically,
            if not it is supposed to be a CLFunctionParameter instance that we append directly.
        cl_header (CLHeaderDefinition): the CL header definition to use. Defaults to CLHeaderFromTemplate.
        cl_code (CLCodeDefinition): the CL code definition to use. Defaults to CLCodeFromAdjacentFile.
        dependency_list (list): the list of functions this function depends on
    """
    name = ''
    description = ''
    cl_function_name = None
    parameter_list = []
    cl_header = CLHeaderFromTemplate()
    cl_code = CLCodeFromAdjacentFile()
    dependency_list = []


class CompartmentBuildingBase(DMRICompartmentModelFunction):
    """Use this class in super calls if you want to overwrite methods in the inherited compartment configs.

    In python2 super needs a type to be able to do its work. This is the type you can give it to allow
    it to do its work.
    """


class CompartmentBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class CompartmentBuildingBase

        Args:
            template (CascadeConfig): the compartment config template to use for creating the class with the right init
                settings.
        """
        class AutoCreatedDMRICompartmentModel(CompartmentBuildingBase):

            def __init__(self, *args):
                new_args = [template.name,
                            template.cl_function_name,
                            _get_parameters_list(template.parameter_list),
                            template.cl_header.get_code(template),
                            template.cl_code.get_code(template),
                            deepcopy(template.dependency_list)]

                for ind, already_set_arg in enumerate(args):
                    new_args[ind] = already_set_arg

                super(AutoCreatedDMRICompartmentModel, self).__init__(*new_args)

                if hasattr(template, 'init'):
                    template.init(self)

        self._bind_functions(template, AutoCreatedDMRICompartmentModel)
        return AutoCreatedDMRICompartmentModel


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
            parameters.append(parameters_loader.load(item))
        else:
            parameters.append(deepcopy(item))
    return parameters


def _construct_cl_function_definition(return_type, cl_function_name, parameters):
    """Create the CL function definition for a compartment function.

    This will construct something like (for the Stick model):
        '''
            mot_float_type cmStick(const mot_float_type4 g,
                                   const mot_float_type b,
                                   const mot_float_type d,
                                   const mot_float_type theta,
                                   const mot_float_type phi)
        '''

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
