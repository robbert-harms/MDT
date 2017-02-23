from mdt.utils import spherical_to_cartesian
from mot.model_building.cl_functions.base import ModelFunction

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
        return '''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {header}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=self._get_cl_dependency_headers(),
                   inclusion_guard_name='DMRICM_' + self.cl_function_name + '_H',
                   header=self._cl_header)

    def get_cl_code(self):
        return '''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=self._get_cl_dependency_code(),
                   inclusion_guard_name='DMRICM_' + self.cl_function_name + '_CL',
                   code=self._cl_code)

    def _get_vector_result_maps(self, theta, phi, vector_name='vec0'):
        """Convert spherical coordinates to cartesian vector in 3d

        Args:
            theta (ndarray): the double array with the theta values
            phi (ndarray): the double array with the phi values
            vector_name (str): the name for this vector, the common naming scheme is:
                <model_name>.<vector_name>[_{0,1,2}]

        Returns:
            dict: containing the cartesian vector with the main the fibre direction.
                It returns only the element .vec0
        """
        cartesian = spherical_to_cartesian(theta, phi)
        extra_dict = {'{}.{}'.format(self.name, vector_name): cartesian}
        return extra_dict
