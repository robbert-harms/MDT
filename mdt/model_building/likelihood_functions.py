from .model_functions import SimpleModelCLFunction
from .parameters import FreeParameter
from mot.library_functions import LogBesseli0
from .parameter_functions.transformations import ClampTransform


__author__ = 'Robbert Harms'
__date__ = "2014-08-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LikelihoodFunction:
    """The likelihood function is the model under which you evaluate the model estimates against observations.

    Since we can have two different versions of the likelihood function (with or without the constant terms),
    this class is a proxy to the model function embedding the likelihood function.
    """

    def get_noise_std_param_name(self):
        """Get the name of the parameter that is associated with the noise standard deviation in the problem data.

        Returns:
            str: the name of the parameter that is associated with the noise_std in the problem data.
        """
        raise NotImplementedError()

    def get_log_likelihood_function(self, include_constant_terms=True):
        """Get the likelihood function as a model function.

        The likelihood function is the log of a Probability Density Function (PDF) which we evaluate for some model
        parameters against some measured data. These likelihood functions typically contain constant terms that are
        unnecessary when the likelihood is used for maximum likelihood estimation. If you do not need the precise
        log likelihood function value, set ``include_constant_terms`` to False, in other cases, set it to True.

        This method should return the log likelihood function as such that when the resulting values are linearly summed
        they would yield the complete log likelihood for the model.

        Furthermore, this should return the log likelihood such that the optimum lies at the maximum. When using
        this log likelihood with a minimization routine, take the negative of this log likelihood.

        Args:
            include_constant_terms (boolean): if we want to include the constant terms or not.

        Returns:
            mdt.model_building.model_functions.ModelCLFunction: The log likelihood function for the given
                observation index under this noise model.
        """
        raise NotImplementedError()


class AbstractLikelihoodFunction(LikelihoodFunction):

    def __init__(self, name, cl_function_name, parameter_list=None, noise_std_param_name=None, dependencies=()):
        """The likelihood model is the model under which you evaluate your model estimates against observations.

        Args:
            name (str): the name of the likelihood model
            cl_function_name (str): the name of the function
            parameter_list (list or tuple): the list of parameters this model requires to function correctly.
                If set to None we set it to a default of three parameters, the observation, the model evaluation and
                the noise std.
            noise_std_param_name (str): the name of the noise sigma parameter
            dependencies (list or tuple): the list of function dependencies
        """
        self._name = name
        self._cl_function_name = cl_function_name
        self._parameter_list = parameter_list or [
            ('double', 'observation'),
            ('double', 'model_evaluation'),
            FreeParameter('mot_float_type', 'sigma', True, 1, 0, 'INFINITY', parameter_transform=ClampTransform())
        ]
        self._noise_std_param_name = noise_std_param_name or 'sigma'
        self._dependencies = dependencies or ()

    def get_noise_std_param_name(self):
        return self._noise_std_param_name

    def get_log_likelihood_function(self, include_constant_terms=True):
        return SimpleModelCLFunction('double', self._cl_function_name, self._parameter_list,
                                     self._get_log_likelihood_body(include_constant_terms),
                                     dependencies=self._dependencies)

    def _get_log_likelihood_body(self, include_constant_terms):
        """Get the CL body for the log likelihood function.

        Args:
            include_constant_terms (boolean): if we want to include the constant terms or not

        Returns:
            str: the function body of the log likelihood model.
        """
        raise NotImplementedError()


class GaussianLikelihoodFunction(AbstractLikelihoodFunction):

    def __init__(self):
        """This uses the log of the Gaussian PDF for the likelihood function.

        The PDF is defined as:

        .. code-block:: c

            PDF = 1/(sigma * sqrt(2*pi)) * exp(-(observation - evaluation)^2 / (2 * sigma^2))

        Taking the log, we get:

        .. code-block:: c

            log(PDF) = - ((observation - evaluation)^2 / (2 * sigma^2)) - log(sigma * sqrt(2*pi))
        """
        super().__init__('GaussianNoiseModel', 'gaussianNoise')

    def _get_log_likelihood_body(self, include_constant_terms):
        if include_constant_terms:
            return '''
                return - pown(observation - model_evaluation, 2) / (2 * sigma * sigma)
                       - log(sigma * sqrt(2 * M_PI));
            '''
        else:
            return '''
                return - pown(observation - model_evaluation, 2) / (2 * sigma * sigma);
            '''


class OffsetGaussianLikelihoodFunction(AbstractLikelihoodFunction):

    def __init__(self):
        """This uses the log of an 'offset Gaussian' PDF for the likelihood function.

        The PDF is defined as:

        .. code-block:: c

            PDF = 1/(sigma * sqrt(2*pi)) * exp(-(observation - sqrt(evaluation^2 + sigma^2))^2 / (2 * sigma^2))

        Where the log of the PDF is given by:

        .. code-block:: c

            log(PDF) = - ((observation - sqrt(evaluation^2 + sigma^2))^2 / (2 * sigma^2)) - log(sigma * sqrt(2*pi))
        """
        super().__init__('OffsetGaussianNoise', 'offsetGaussian')

    def _get_log_likelihood_body(self, include_constant_terms):
        if include_constant_terms:
            return '''
                return - pown(observation - hypot(model_evaluation, (double)sigma), 2) / (2 * sigma * sigma)
                       - log(sigma * sqrt(2 * M_PI));
            '''
        else:
            return '''
                return - pown(observation - hypot(model_evaluation, (double)sigma), 2) / (2 * sigma * sigma);
            '''


class RicianLikelihoodFunction(AbstractLikelihoodFunction):

    def __init__(self):
        """This uses the log of the Rice PDF for the likelihood function.

        The PDF is defined as:

        .. code-block:: c

            PDF = (observation/sigma^2)
                    * exp(-(observation^2 + evaluation^2) / (2 * sigma^2))
                    * bessel_i0((observation * evaluation) / sigma^2)

        Where where ``bessel_i0(z)`` is the modified Bessel function of the first kind with order zero.

        The log of the PDF is given by:

        .. code-block:: c

            log(PDF) = log(observation/sigma^2)
                        - (observation^2 + evaluation^2) / (2 * sigma^2)
                        + log(bessel_i0((observation * evaluation) / sigma^2))
        """
        super().__init__('RicianNoise', 'ricianNoise', dependencies=(LogBesseli0(),))

    def _get_log_likelihood_body(self, include_constant_terms):
        if include_constant_terms:
            return '''
                return - ((model_evaluation * model_evaluation) / (2 * sigma * sigma))
                       + log_bessel_i0((observation * model_evaluation) / (sigma * sigma))
                       + log(observation / (sigma * sigma))
                       - ((observation * observation) / (2 * sigma * sigma));
            '''
        else:
            return '''
                return  - ((model_evaluation * model_evaluation) / (2 * sigma * sigma))
                        + log_bessel_i0((observation * model_evaluation) / (sigma * sigma));
            '''
