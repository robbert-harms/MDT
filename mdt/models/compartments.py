from textwrap import dedent, indent

from mot.model_building.model_functions import SimpleModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-12-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICompartmentModelFunction(SimpleModelFunction):

    def __init__(self, name, cl_function_name, parameter_list, cl_code, dependency_list, return_type,
                 model_function_priors=None, post_optimization_modifiers=None,
                 sampling_covar_extras=None, sampling_covar_exclude=None):
        """Create a new dMRI compartment model function.

        Args:
            name (str): the name of this compartment model
            cl_function_name (str): the name of this function in the CL kernel
            parameter_list (list of CLFunctionParameter): the list of the function parameters
            cl_code (str): the code for the function in CL
            dependency_list (list): the list of functions we depend on inside the kernel
            return_type (str): the CL return type
            model_function_priors (list of mot.model_building.model_function_priors.ModelFunctionPrior): additional
                compartment priors on top of the parameter priors.
            post_optimization_modifiers (None or list or tuple): a list of modification callbacks for use after
                optimization. Examples:

                .. code-block:: python

                    post_optimization_modifiers = [('vec0', lambda d: spherical_to_cartesian(d['theta'], d['phi'])),
                                                   ...]

                These modifiers are supposed to be called before the post optimization modifiers of the composite model.

            sampling_covar_extras (None or tuple or list): information about a single callback function that can
            add additional maps to the covariance matrix calculated after sampling. Usage example::

                sampling_covar_extras = (('theta', 'phi'), ('vec0_x', 'vec0_y', 'vec0_z'), spherical_to_cartesian)

            sampling_covar_exclude (None tuple or list): parameters to exclude in the covariance matrix calculation
                after sampling. Example::

                    sampling_covar_exclude = ['theta', 'phi']

        """
        super(DMRICompartmentModelFunction, self).__init__(return_type, name, cl_function_name, parameter_list,
                                                           dependency_list=dependency_list,
                                                           model_function_priors=model_function_priors)
        self._cl_code = cl_code
        self.post_optimization_modifiers = post_optimization_modifiers or []
        self.sampling_covar_extras = sampling_covar_extras
        self.sampling_covar_exclude = sampling_covar_exclude

    def get_cl_code(self):
        return dedent('''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=indent(self._get_cl_dependency_code(), ' '*4*3),
                   inclusion_guard_name='DMRI_' + self.cl_function_name + '_CL',
                   code=indent('\n' + self._cl_code.strip() + '\n', ' '*4*3)))
