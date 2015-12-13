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

    def __init__(self, name, cl_function_name, parameter_list, cl_header_file, cl_code_file, dependency_list):
        super(DMRICompartmentModelFunction, self).__init__(name, cl_function_name, parameter_list,
                                                           dependency_list=dependency_list)
        self._cl_header_file = cl_header_file
        self._cl_code_file = cl_code_file

    def get_cl_header(self):
        inclusion_guard_name = 'DMRICM_' + os.path.splitext(os.path.basename(self._cl_header_file))[0] + '_H'

        header = self._get_cl_dependency_headers() + "\n"
        header += '''
            #ifndef {0}
            #define {0}
        '''.format(inclusion_guard_name)
        header += open(os.path.abspath(self._cl_header_file), 'r').read()
        header += '''
            #endif // {0}
        '''.format(inclusion_guard_name)

        return header

    def get_cl_code(self):
        inclusion_guard_name = 'DMRICM_' + os.path.splitext(os.path.basename(self._cl_header_file))[0] + '_CL'

        code = self._get_cl_dependency_code() + "\n"
        code += '''
            #ifndef {0}
            #define {0}
        '''.format(inclusion_guard_name)
        code += open(os.path.abspath(self._cl_code_file), 'r').read()
        code += '''
            #endif // {0}
        '''.format(inclusion_guard_name)
        return code

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
        cl_code_file (str): the full path to the CL code file. You don't need to define this if you set
            module_name=__name__ in your config dict
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
        cl_code_file=None,
        dependency_list=[],
        module_name=None
    )

    def __init__(self, *args, **kwargs):
        new_args = [self.get_config_attribute('name'),
                    self.get_config_attribute('cl_function_name'),
                    self.get_parameters_list(),
                    self.get_cl_header_file_name(),
                    self.get_cl_code_file_name(),
                    self.get_config_attribute('dependency_list')]

        for ind, already_set_arg in enumerate(args):
            new_args[ind] = already_set_arg

        super(DMRICompartmentModelBuilder, self).__init__(*new_args)

    @classmethod
    def get_parameters_list(cls):
        parameters = []
        for item in cls.get_config_attribute('parameter_list'):
            if isinstance(item, six.string_types):
                parameters.append(get_parameter(item))
            else:
                parameters.append(deepcopy(item))
        return parameters

    @classmethod
    def get_cl_header_file_name(cls):
        if cls.get_config_attribute('cl_header_file') is None:
            return resource_filename(cls.get_config_attribute('module_name'), cls.get_config_attribute('name') + '.h')
        return cls.get_config_attribute('cl_header_file')

    @classmethod
    def get_cl_code_file_name(cls):
        if cls.get_config_attribute('cl_code_file') is None:
            return resource_filename(cls.get_config_attribute('module_name'), cls.get_config_attribute('name') + '.cl')
        return cls.get_config_attribute('cl_code_file')

    @classmethod
    def meta_info(cls):
        return {'name': cls.get_config_attribute('name'),
                'description': cls.get_config_attribute('description')}

    @classmethod
    def get_config_attribute(cls, name):
        return cls.config.get(name, cls.config_default[name])
