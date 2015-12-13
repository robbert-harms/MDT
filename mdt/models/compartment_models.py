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
        return self._get_cl_dependency_code() + "\n" + open(os.path.abspath(self._cl_code_file), 'r').read()

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
        parameter_list (
    """
    config = {}
    config_default = dict(
        name='',
        description='',
        cl_function_name=None,
        parameter_list=[],
        cl_header_file=None,
        cl_code_file=None,
        dependency_list=[]
    )

    def __init__(self, *args):
        new_args = [self.get_config_attribute('name'),
                    self.get_config_attribute('cl_function_name'),
                    self.get_parameters_list(),
                    self.get_config_attribute('cl_header_file'),
                    self.get_config_attribute('cl_code_file'),
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
    def meta_info(cls):
        return {'name': cls.get_config_attribute('name'),
                'description': cls.get_config_attribute('description')}

    @classmethod
    def get_config_attribute(cls, name):
        return cls.config.get(name, cls.config_default[name])
