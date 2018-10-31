from mot.lib.cl_function import SimpleCLFunctionParameter
from .parameter_functions.numdiff_info import SimpleNumDiffInfo
from .parameter_functions.priors import UniformWithinBoundsPrior
from .parameter_functions.transformations import IdentityTransform

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class InputDataParameter(SimpleCLFunctionParameter):

    def __init__(self, declaration, value):
        """These parameters signal are meant to be contain data loaded from the input data object.

        In contrast to free parameters which are being optimized (or fixed to values), these parameters are
        meant to be loaded from the input data. They can contain scalars, vectors or matrices with values
        to use for each problem instance and each data point.

        Args:
            declaration (str): the declaration of this parameter. For example ``global int foo``.
            value (double or ndarray): The value used if no value is given in the input data.
        """
        super().__init__(declaration)
        self.value = value


class ProtocolParameter(InputDataParameter):

    def __init__(self, declaration, value=None):
        """Caries data per observation.

        Values for this parameter type are typically loaded from the input data. A default can be provided in the case
        that there is no suitable value in the input data.

        Args:
            declaration (str): the declaration of this parameter. For example ``global int foo``.
            value (None or float or ndarray): The value used if no value is given in the input data.
        """
        super().__init__(declaration, value=value)


class FreeParameter(SimpleCLFunctionParameter):

    def __init__(self, declaration, fixed, value, lower_bound, upper_bound,
                 parameter_transform=None, sampling_proposal_std=None,
                 sampling_prior=None, numdiff_info=None):
        """This are the kind of parameters that are generally meant to be optimized.

        Args:
            declaration (str): the declaration of this parameter. For example ``global int foo``.
            fixed (boolean): If this parameter is fixed to the value given
            value (double or ndarray): A single value for all problems or a list of values for each problem.
            lower_bound (double): The lower bound of this parameter
            upper_bound (double): The upper bound of this parameter
            parameter_transform (mdt.model_building.parameter_functions.transformations.AbstractTransformation):
                The parameter transformation function
            sampling_proposal_std (float): The proposal standard deviation, used in some MCMC sample routines
            sampling_prior (mdt.model_building.parameter_functions.priors.ParameterPrior): The prior function for
                use in model sample
            numdiff_info (mdt.model_building.parameter_functions.numdiff_info.NumDiffInfo): the information
                for taking the numerical derivative with respect to this parameter.
        """
        super().__init__(declaration)
        self._value = value
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._fixed = fixed

        self._parameter_transform = parameter_transform or IdentityTransform()
        self._sampling_proposal_std = sampling_proposal_std or 1
        self._sampling_prior = sampling_prior or UniformWithinBoundsPrior()
        self._numdiff_info = numdiff_info or SimpleNumDiffInfo()

    @property
    def value(self):
        return self._value

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def fixed(self):
        return self._fixed

    @property
    def parameter_transform(self):
        """Get the parameter transformation function used during optimization.

        Returns:
            mdt.model_building.parameter_functions.transformations.AbstractTransformation: the transformation method
        """
        return self._parameter_transform

    @property
    def sampling_proposal_std(self):
        """Get the initial proposal standard deviation for this parameter.

        Returns:
            float: the initial default proposal standard deviation for use in MCMC sampling
        """
        return self._sampling_proposal_std

    @property
    def sampling_prior(self):
        """Get the prior for this parameter, this is used in MCMC sampling.

        Returns:
            mdt.model_building.parameter_functions.priors.ParameterPrior: the prior for this parameter
        """
        return self._sampling_prior

    @property
    def numdiff_info(self):
        """Specifies how to numerically differentiate this parameter.

        Returns:
            mdt.model_building.parameter_functions.numdiff_info.NumDiffInfo: the numerical differentiation information
        """
        return self._numdiff_info


class NoiseStdFreeParameter(FreeParameter):

    def __init__(self, *args, **kwargs):
        """Specifies that this parameter should be set to the current noise standard deviation estimate.

        Parameters of this type are only meant to be used in the likelihood functions. They indicate the
        parameter to use for the initialization of the noise standard deviation.
        """
        super().__init__(*args, **kwargs)


class NoiseStdInputParameter(SimpleCLFunctionParameter):

    def __init__(self, name='noise_std'):
        """Parameter indicating that this parameter should be fixed to the current value of the noise std.

        Parameters of this type are meant to be used in compartment models specifying that we should use the
        noise standard deviation value from the likelihood function as input to this function.
        """
        super().__init__('double ' + name)


class LibraryParameter(SimpleCLFunctionParameter):
    """Parameters of this type are used inside library functions. They are not meant to be used in Model functions.
    """


class CurrentObservationParam(SimpleCLFunctionParameter):

    def __init__(self, name='observation'):
        """This parameter indicates that the model should inject the current observation value in the model.

        Sometimes during model linearization or other mathematical operations the current observation appears on
        both sides of the optimization equation. That is, it sometimes happens you want to use the current observation
        to model that same observation. This parameter is a signal to the model builder to inject the current
        observation.

        You can use this parameter by adding it to your model and then use the current name in your model equation.
        """
        super().__init__('double ' + name)


class CurrentModelSignalParam(SimpleCLFunctionParameter):

    def __init__(self, name='model_signal'):
        """This parameter indicates that the model should inject here the current signal value of the model.

        Parameters of this type can only be used by the signal noise and the likelihood functions.
        """
        super().__init__('double ' + name)


class DataCacheParameter(SimpleCLFunctionParameter):

    def __init__(self, compartment_name, name):
        """This class provides a subclass for checking instance types.

        Args:
            compartment_name (str): the name of the compartment holding this parameter.
                This parameter will make sure it gets named to the correct caching struct type.
            name (str): the name of this parameter in the function
        """
        super().__init__('{}_cache* {}'.format(compartment_name, name))
