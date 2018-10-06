import numpy as np
from mot.lib.cl_function import CLFunction, SimpleCLFunction


__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterPrior(CLFunction):
    """The priors are used during model sample, indicating the a priori information one has about a parameter.

    These priors are not in log space, we take the log in the model builder.

    The signature of prior parameters must be:

    .. code-block: c

            mot_float_type <prior_fname>(mot_float_type value,
                                         mot_float_type lower_bound,
                                         mot_float_type upper_bound,
                                         <extra_parameters>)
    """

    def get_extra_parameters(self):
        """Get the additional prior specific parameters.

        Each prior has at least 3 parameters, the value, lower bound and upper bound, but it can have more parameters.

        Returns:
            list: list of additional parameters
        """
        raise NotImplementedError()


class SimpleParameterPrior(ParameterPrior, SimpleCLFunction):

    def __init__(self, prior_name, prior_body, extra_params=None, dependencies=()):
        """A prior template function.

        This will prepend to the given extra parameters the obligatory parameters (value, lower_bound, upper_bound).

        Args:
            prior_name (str): the name of this prior function
            prior_body (str): the body of the prior
            extra_params (list): additional parameters for this prior
            dependencies (list or tuple): the list of dependency functions
        """
        self.extra_params = extra_params or []
        parameters = [('mot_float_type', 'value'),
                      ('mot_float_type', 'lower_bound'),
                      ('mot_float_type', 'upper_bound')] + self.extra_params
        super().__init__('mot_float_type', prior_name, parameters, prior_body, dependencies=dependencies)

    def get_extra_parameters(self):
        return self.extra_params


class AlwaysOne(SimpleParameterPrior):

    def __init__(self):
        """The uniform prior is always 1. :math:`P(v) = 1` """
        super().__init__('uniform', 'return 1;')


class ReciprocalPrior(SimpleParameterPrior):

    def __init__(self):
        """The reciprocal of the current value. :math:`P(v) = 1/v` """
        body = '''
            if(value <= 0){
                return 0;
            }
            return 1.0/value;
        '''
        super().__init__('reciprocal', body)


class UniformWithinBoundsPrior(SimpleParameterPrior):

    def __init__(self):
        """This prior is 1 within the upper and lower bound of the parameter, 0 outside."""
        super().__init__(
            'uniform_within_bounds',
            'return value >= lower_bound && value <= upper_bound;')


class AbsSinPrior(SimpleParameterPrior):

    def __init__(self):
        """Angular prior: :math:`P(v) = |\\sin(v)|`"""
        super().__init__('abs_sin', 'return fabs(sin(value));')


class AbsSinHalfPrior(SimpleParameterPrior):

    def __init__(self):
        """Angular prior: :math:`P(v) = |\\sin(x)/2.0|`"""
        super().__init__('abs_sin_half', 'return fabs(sin(value)/2.0);')


class VagueGammaPrior(SimpleParameterPrior):

    def __init__(self):
        """The vague gamma prior is meant as a proper uniform prior.

        Lee & Wagenmakers:

            The practice of assigning Gamma(0.001, 0.001) priors on precision parameters is theoretically motivated by
            scale invariance arguments, meaning that priors are chosen so that changing the measurement
            scale of the data does not affect inference.
            The invariant prior on precision lambda corresponds to a uniform distribution on log sigma,
            that is, rho(sigma^2) prop. to. 1/sigma^2, or a Gamma(a -> 0, b -> 0) distribution.
            This invariant prior distribution, however, is improper (i.e., the area under the curve is unbounded),
            which means it is not really a distribution, but the limit of a sequence of continuous_distributions
            (see Jaynes, 2003). WinBUGS requires the use of proper continuous_distributions,
            and the Gamma(0.001, 0.001) prior is intended as a proper approximation to the theoretically
            motivated improper prior. This raises the issue of whether inference is sensitive to the essentially
            arbitrary value 0.001, and it is sometimes the case that using other small values such as 0.01 or 0.1
            leads to more stable sample
            in WinBUGS.

            -- Lee & Wagenmakers, Bayesian Cognitive Modeling, 2014, Chapter 4, Box 4.1

        While this is not WinBUGS and improper priors are allowed in MOT, it is still useful to have this prior
        in case people desire proper priors.
        """
        body = '''
            float kappa = 0.001;
            float theta = 1/0.001;

            return (1.0 / (tgamma(kappa) * pow(theta, kappa))) * pow(value, kappa - 1) * exp(- value / theta);
        '''
        super().__init__('vague_gamma_prior', body)


class NormalPDF(SimpleParameterPrior):

    def __init__(self):
        r"""Normal PDF on the given value: :math:`P(v) = N(v; \mu, \sigma)`"""
        from mdt.model_building.parameters import FreeParameter
        extra_params = [FreeParameter('mot_float_type', 'mu', True, 0, -np.inf, np.inf,
                                      sampling_prior=AlwaysOne()),
                        FreeParameter('mot_float_type', 'sigma', True, 1, -np.inf, np.inf,
                                      sampling_prior=AlwaysOne())]

        super().__init__(
            'normal_pdf',
            'return exp((mot_float_type) (-((value - mu) * (value - mu)) / (2 * sigma * sigma))) '
            '           / (sigma * sqrt(2 * M_PI));',
            extra_params)


class AxialNormalPDF(SimpleParameterPrior):

    def __init__(self):
        r"""The axial normal PDF is a Normal distribution wrapped around 0 and :math:`\pi`.

        It's PDF is given by:

        .. math::

            f(\theta; a, b) = \frac{\cosh(a\sin \theta + b\cos \theta)}{\pi I_{0}(\sqrt{a^{2} + b^{2}})}

        where in this implementation :math:`a` and :math:`b` are parameterized with the input variables
        :math:`\mu` and :math:`\sigma` using:

        .. math::

            \begin{align*}
            \kappa &= \frac{1}{\sigma^{2}} \\
            a &= \kappa * \sin \mu \\
            b &= \kappa * \cos \mu
            \end{align*}

        References:
            Barry C. Arnold, Ashis SenGupta (2006). Probability continuous_distributions and statistical inference for axial data.
            Environmental and Ecological Statistics, volume 13, issue 3, pages 271-285.
        """
        from mdt.model_building.parameters import FreeParameter
        from mot.library_functions import LogCosh, LogBesseli0

        extra_params = [FreeParameter('mot_float_type', 'mu', True, 0, -np.inf, np.inf,
                                      sampling_prior=AlwaysOne()),
                        FreeParameter('mot_float_type', 'sigma', True, 1, -np.inf, np.inf,
                                      sampling_prior=AlwaysOne())]

        super().__init__(
            'axial_normal_pdf',
            '''
                float kappa = 1.0 / pown(sigma, 2);
                float a = kappa * sin(mu);
                float b = kappa * cos(mu);

                return exp(log_cosh(a * sin(value) + b * cos(value))
                            - log_bessel_i0(sqrt(pown(a, 2) + pown(b, 2)))
                            - log(M_PI) );
            ''',
            extra_params,
            dependencies=(LogBesseli0(), LogCosh()))


class ARDBeta(SimpleParameterPrior):

    def __init__(self):
        r"""This is a collapsed form of the Beta PDF meant for use in Automatic Relevance Detection sampling.

        In this prior the ``alpha`` parameter of the Beta prior is set to 1 which simplifies the equation.
        The parameter ``beta`` is still free and can be changed as desired.

        The implemented prior is:

        .. math::

            B(x; 1, \beta) = \beta * (1 - x)^{\beta - 1}

        """
        from mdt.model_building.parameters import FreeParameter
        extra_params = [FreeParameter('mot_float_type', 'beta', False, 1, 1e-4, 1000,
                                      sampling_prior=ReciprocalPrior(),
                                      sampling_proposal_std=0.01)]

        body = '''
            if(value < lower_bound || value > upper_bound){
                return 0;
            }
            return beta * pow(1 - value, beta - 1);
        '''
        super().__init__('ard_beta_pdf', body, extra_params)


class ARDGaussian(SimpleParameterPrior):

    def __init__(self):
        """This is a Gaussian prior meant for use in Automatic Relevance Detection sampling.

        This uses a Gaussian prior with mean at zero and a standard deviation determined by the ``alpha`` parameter
        with the relationship :math:`\sigma = 1/\\sqrt(\\alpha)`.
        """
        from mdt.model_building.parameters import FreeParameter
        extra_params = [FreeParameter('mot_float_type', 'alpha', False, 8, 1e-5, 1e4,
                                      sampling_prior=UniformWithinBoundsPrior(),
                                      sampling_proposal_std=1)]

        body = '''
            if(value < lower_bound || value > upper_bound){
                return 0;
            }
            mot_float_type sigma = 1.0/sqrt(alpha);
            return exp(-pown(value, 2) / (2 * pown(sigma, 2))) / (sigma * sqrt(2 * M_PI));
        '''
        super().__init__('ard_gaussian_pdf', body, extra_params)
