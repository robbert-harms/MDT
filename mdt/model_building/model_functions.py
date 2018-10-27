from mot.lib.cl_function import CLFunction, SimpleCLFunction
from .parameter_functions.numdiff_info import SimpleNumDiffInfo
from .parameters import FreeParameter
from .parameter_functions.priors import UniformWithinBoundsPrior
from .parameter_functions.transformations import CosSqrClampTransform

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelCLFunction(CLFunction):
    """Extends a CLFunction with modeling information."""

    @property
    def name(self):
        """Get the name of this model function.

        Returns:
            str: The name of this model function.
        """
        raise NotImplementedError()

    def get_free_parameters(self):
        """Get all the free parameters in this model

        Returns:
            list of CLFunctionParameter: list of all the model parameters of type FreeParameter in this model
        """
        raise NotImplementedError()

    def get_prior_parameters(self, parameter):
        """Get the prior parameters of the given parameter.

        Args:
            parameter (FreeParameter): one of the parameters of this model function

        Returns:
            list of parameters: the list of prior parameters for the given parameter
        """
        raise NotImplementedError()

    def get_model_function_priors(self):
        """Get all the model function priors.

        Returns:
            list of mot.lib.cl_function.CLFunction: the priors for this model function,
                these are supposed to be used in conjunction to the parameter priors.
        """
        raise NotImplementedError()


class SimpleModelCLFunction(ModelCLFunction, SimpleCLFunction):

    def __init__(self, return_type, cl_function_name, parameters, cl_body, dependencies=None,
                 model_function_priors=None):
        """A default implementation of a This CL function is for all estimable models

        Args:
            return_type (str): the CL return type of the function
            cl_function_name (string): The name of the CL function
            parameters (list or tuple of CLFunctionParameter): The list of parameters required for this function
            cl_body (str): the cl body of this function
            dependencies (list or tuple of CLFunction): The list of CL libraries this function depends on
            model_function_priors (list of mot.lib.cl_function.CLFunction): list of priors concerning this whole model
                function. The parameter names of the given functions must match those of this function.
        """
        super().__init__(return_type, cl_function_name, parameters,
                         cl_body, dependencies=dependencies)
        self._model_function_priors = model_function_priors or []
        if isinstance(self._model_function_priors, CLFunction):
            self._model_function_priors = [self._model_function_priors]

    @property
    def name(self):
        """Get the name of this model function, used in the composite model function

        Returns:
            str: The name of this model function.
        """
        return self.get_cl_function_name()

    def get_model_function_priors(self):
        """Get all the model function priors.

        Returns:
            list[mot.lib.cl_function.CLFunction]: the priors for this model function, these are supposed to be used in
                conjunction to the parameter priors.
        """
        return self._model_function_priors

    def get_free_parameters(self):
        """Get all the free parameters in this model

        Returns:
            list: the list of free parameters in this model
        """
        return list([p for p in self.get_parameters() if isinstance(p, FreeParameter)])

    def get_prior_parameters(self, parameter):
        """Get the parameters referred to by the priors of each of the free parameters.

        This returns a list of all the parameters referenced by the priors of the parameters, recursively.

        Returns:
            list of mot.lib.cl_function.CLFunctionParameter: the list of additional parameters used by each of the
                parameter priors
        """
        def get_prior_parameters(params):
            return_params = []

            for param in params:
                prior_params = param.sampling_prior.get_extra_parameters()
                proxy_prior_params = [prior_param.get_renamed('{}.prior.{}'.format(param.name, prior_param.name))
                                      for prior_param in prior_params]

                return_params.extend(proxy_prior_params)

                free_prior_params = [p for p in proxy_prior_params if isinstance(p, FreeParameter)]
                return_params.extend(get_prior_parameters(free_prior_params))

            return return_params

        return get_prior_parameters([parameter])


class WeightType(SimpleModelCLFunction):
    """A class that defines the notion of a weighted compartment.

    Some of the code checks for this class type, be sure to subclass this class if you want to represent a Weight.
    """


class SimpleWeight(WeightType):

    def __init__(self, name='Weight', param_name='w', value=0.5, lower_bound=0.0,
                 upper_bound=1.0, parameter_kwargs=None):
        """A class that by itself defines the notion of a Weight.

        Some of the code checks for type Weight, be sure to use this model function if you want to represent a Weight.

        A weight is meant to be a model volume fraction.

        Args:
            name (str): The name of the model
            value (number or ndarray): The initial value for the single free parameter of this function.
            lower_bound (number or ndarray): The initial lower bound for the single free parameter of this function.
            upper_bound (number or ndarray): The initial upper bound for the single free parameter of this function.
        """
        parameter_settings = dict(parameter_transform=CosSqrClampTransform(),
                                  sampling_proposal_std=0.01,
                                  sampling_prior=UniformWithinBoundsPrior(),
                                  numdiff_info=SimpleNumDiffInfo(scale_factor=10)
                                  )
        parameter_settings.update(parameter_kwargs or {})

        super().__init__(
            'mot_float_type',
            name,
            (FreeParameter('mot_float_type ' + param_name, False, value,
                           lower_bound, upper_bound, **parameter_settings),),
            'return ' + param_name + ';')
