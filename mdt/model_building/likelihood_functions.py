import numpy as np
from .model_functions import SimpleModelCLFunction
from .parameters import CurrentObservationParam, CurrentModelSignalParam, NoiseStdFreeParameter
from mot.library_functions import LogBesseli0, normal_logpdf
from .parameter_functions.transformations import ClampTransform


__author__ = 'Robbert Harms'
__date__ = "2014-08-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LikelihoodFunction(SimpleModelCLFunction):
    """The likelihood function is the model under which you evaluate the model estimates against observations."""


class Gaussian(LikelihoodFunction):

    def __init__(self):
        """This uses the log of the Gaussian PDF for the likelihood function.

        The PDF is defined as:

        .. code-block:: c

            PDF = 1/(sigma * sqrt(2*pi)) * exp(-(observation - evaluation)^2 / (2 * sigma^2))

        Taking the log, we get:

        .. code-block:: c

            log(PDF) = - ((observation - evaluation)^2 / (2 * sigma^2)) - log(sigma * sqrt(2*pi))
        """
        parameter_list = [
            CurrentObservationParam('observation'),
            CurrentModelSignalParam('model_evaluation'),
            NoiseStdFreeParameter('mot_float_type sigma', True, 1, 0, np.inf, parameter_transform=ClampTransform())
        ]
        body = '''
            return normal_logpdf(observation, model_evaluation, sigma);
        '''
        super().__init__('double', 'Gaussian', parameter_list, body, dependencies=(normal_logpdf(),))


class OffsetGaussian(LikelihoodFunction):

    def __init__(self):
        """This uses the log of an 'offset Gaussian' PDF for the likelihood function.

        The PDF is defined as:

        .. code-block:: c

            PDF = 1/(sigma * sqrt(2*pi)) * exp(-(observation - sqrt(evaluation^2 + sigma^2))^2 / (2 * sigma^2))

        Where the log of the PDF is given by:

        .. code-block:: c

            log(PDF) = - ((observation - sqrt(evaluation^2 + sigma^2))^2 / (2 * sigma^2)) - log(sigma * sqrt(2*pi))
        """
        parameter_list = [
            CurrentObservationParam('observation'),
            CurrentModelSignalParam('model_evaluation'),
            NoiseStdFreeParameter('mot_float_type sigma', True, 1, 0, np.inf, parameter_transform=ClampTransform())
        ]
        body = '''
            return normal_logpdf(observation, hypot(model_evaluation, (double)sigma), sigma);
        '''
        super().__init__('double', 'OffsetGaussian', parameter_list, body, dependencies=(normal_logpdf(),))


class Rician(LikelihoodFunction):

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
        parameter_list = [
            CurrentObservationParam('observation'),
            CurrentModelSignalParam('model_evaluation'),
            NoiseStdFreeParameter('mot_float_type sigma', True, 1, 0, np.inf, parameter_transform=ClampTransform())
        ]
        body = '''
            double obs_div = observation / sigma;
            double eval_div = model_evaluation / sigma;
            
            return   log(obs_div / sigma)
                   - ((obs_div * obs_div + eval_div * eval_div) / 2)
                   + log_bessel_i0(obs_div * eval_div);
        '''
        super().__init__('double', 'Rician', parameter_list, body, dependencies=(LogBesseli0(),))
