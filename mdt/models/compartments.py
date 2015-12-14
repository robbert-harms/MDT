from pkg_resources import resource_filename
import os
from copy import deepcopy
import six
from mdt.model_parameters import get_parameter
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

    def _get_single_dir_coordinate_maps(self, theta, phi, r):
        """Convert spherical coordinates to cartesian coordinates in 3d

        Args:
            theta (ndarray): the double array with the theta values
            phi (ndarray): the double array with the phi values
            r (ndarray): the double array with the r values

        Returns:
            three ndarrays, per vector one map
        """
        cartesian = spherical_to_cartesian(theta, phi)
        extra_dict = {self.name + '.eig0.vec': cartesian, self.name + '.eig0.val': r}

        for ind in range(3):
            extra_dict.update({self.name + '.eig0.vec.' + repr(ind): cartesian[:, ind]})

        return extra_dict


class DMRICompartmentModelBuilder(DMRICompartmentModelFunction):
    """The compartment model builder to inherit from.

    One can use this to create models in a declarative style. This works because in the constructor we use deepcopy
    to copy all the relevant material before creating a new instance of the class.

    Class attributes:
        name (str): the name of the model
        description (str): model description
        cl_function_name (str): the name of the function in the CL kernel
        parameter_list (list): the list of parameters to use. If a parameter is a string we will load it automatically,
            if not it is supposed to be a CLFunctionParameter instance that we append directly.
        cl_header_file (str): the full path to the CL header file. You don't need to define this if you set
            module_name = __name__ in your config dict
        cl_header (str): the CL header. This is only used if cl_header_file is not set and or no suitable
            header file could be loaded. If not defined we auto-create one.
        cl_code_file (str): the full path to the CL code file. You don't need to define this if you set
            module_name=__name__ in your config dict
        cl_code (str): the CL code to use. This is only used if cl_code_file is not set and or no suitable
            code file could be loaded. For simple functions you can also use cl_code_inline.
        cl_code_inline (str): the inline CL code. This is meant for simple functions for which we can autoconstruct
            the function signature and inline this code.
        dependency_list (list): the list of functions this function depends on
        module_name (str): the name of the module implementing the subclass. You always need to set this to __name__:
            module_name=__name__
    """
    config = {}
    config_default = dict(
        name='',
        description='',
        cl_function_name=None,
        parameter_list=[],
        cl_header_file=None,
        cl_header=None,
        cl_code_file=None,
        cl_code=None,
        cl_code_inline=None,
        dependency_list=[],
        module_name=None
    )

    def __init__(self, *args, **kwargs):
        new_args = [self._get_config_attribute('name'),
                    self._get_config_attribute('cl_function_name'),
                    self._get_parameters_list(),
                    self._get_cl_header_from_config(),
                    self._get_cl_code_from_config(),
                    self._get_config_attribute('dependency_list')]

        for ind, already_set_arg in enumerate(args):
            new_args[ind] = already_set_arg

        super(DMRICompartmentModelBuilder, self).__init__(*new_args)

    @classmethod
    def meta_info(cls):
        return {'name': cls._get_config_attribute('name'),
                'description': cls._get_config_attribute('description')}

    @classmethod
    def _get_parameters_list(cls):
        parameters = []
        for item in cls._get_config_attribute('parameter_list'):
            if isinstance(item, six.string_types):
                parameters.append(get_parameter(item))
            else:
                parameters.append(deepcopy(item))
        return parameters

    @classmethod
    def _get_cl_header_from_config(cls):
        if cls._get_config_attribute('cl_header_file'):
            open(os.path.abspath(cls._get_config_attribute('cl_header_file')), 'r').read()

        if cls._get_config_attribute('module_name'):
            path = os.path.abspath(resource_filename(cls._get_config_attribute('module_name'),
                                                     cls._get_config_attribute('name') + '.h'))
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    return f.read()

        if cls._get_config_attribute('cl_header'):
            return cls._get_config_attribute('cl_header')

        return _construct_cl_function_definition('MOT_FLOAT_TYPE',
                                                 cls._get_config_attribute('cl_function_name'),
                                                 cls._get_parameters_list()) + ';'

    @classmethod
    def _get_cl_code_from_config(cls):
        if cls._get_config_attribute('cl_code_file'):
            open(os.path.abspath(cls._get_config_attribute('cl_code_file')), 'r').read()

        if cls._get_config_attribute('module_name'):
            path = os.path.abspath(resource_filename(cls._get_config_attribute('module_name'),
                                                     cls._get_config_attribute('name') + '.cl'))
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    return f.read()

        if cls._get_config_attribute('cl_code'):
            return cls._get_config_attribute('cl_code')

        s = _construct_cl_function_definition('MOT_FLOAT_TYPE',
                                              cls._get_config_attribute('cl_function_name'),
                                              cls._get_parameters_list())
        s += '{\n' + cls._get_config_attribute('cl_code_inline') + '\n}'
        return s

    @classmethod
    def _get_config_attribute(cls, name):
        return cls.config.get(name, cls.config_default[name])


def _construct_cl_function_definition(return_type, cl_function_name, parameters):
    """Create the CL function definition for a compartment function.

    This will construct something like (for the Stick model):
        '''
            MOT_FLOAT_TYPE cmStick(const MOT_FLOAT_TYPE4 g,
                                   const MOT_FLOAT_TYPE b,
                                   const MOT_FLOAT_TYPE d,
                                   const MOT_FLOAT_TYPE theta,
                                   const MOT_FLOAT_TYPE phi)
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
