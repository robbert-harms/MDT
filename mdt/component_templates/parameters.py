import numpy as np
from mdt.component_templates.base import ComponentBuilder, ComponentTemplate
from mdt.lib.components import has_component, get_component
from mdt.model_building.parameter_functions.numdiff_info import NumDiffInfo, SimpleNumDiffInfo
from mdt.model_building.parameters import ProtocolParameter, FreeParameter
from mdt.model_building.parameter_functions.priors import UniformWithinBoundsPrior
from mdt.model_building.parameter_functions.transformations import AbstractTransformation


__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterBuilder(ComponentBuilder):

    def _create_class(self, template):
        """Creates classes with as base class DMRICompositeModel

        Args:
            template (Type[ParameterTemplate]): the configuration for the parameter.
        """
        if issubclass(template, ProtocolParameterTemplate):
            class AutoProtocolParameter(ProtocolParameter):
                def __init__(self, nickname=None):
                    super().__init__('{} {}'.format(template.data_type, nickname or template.name),
                                     value=template.value)

            for name, method in template.bound_methods.items():
                setattr(AutoProtocolParameter, name, method)

            return AutoProtocolParameter

        elif issubclass(template, FreeParameterTemplate):
            numdiff_info = template.numdiff_info
            if not isinstance(numdiff_info, NumDiffInfo) and numdiff_info is not None:
                numdiff_info = SimpleNumDiffInfo(**numdiff_info)

            class AutoFreeParameter(FreeParameter):
                def __init__(self, nickname=None):
                    super().__init__(
                        '{} {}'.format(template.data_type, nickname or template.name),
                        template.fixed,
                        template.init_value,
                        template.lower_bound,
                        template.upper_bound,
                        parameter_transform=_resolve_parameter_transform(template.parameter_transform),
                        sampling_proposal_std=template.sampling_proposal_std,
                        sampling_prior=template.sampling_prior,
                        numdiff_info=numdiff_info
                    )
                    self.sampling_proposal_modulus = template.sampling_proposal_modulus

            for name, method in template.bound_methods.items():
                setattr(AutoFreeParameter, name, method)

            return AutoFreeParameter


class ParameterTemplate(ComponentTemplate):
    """The cascade template to inherit from.

    These templates are loaded on the fly by the ParametersBuilder

    template options:
        name (str): the name of the parameter, defaults to the class name
        description (str): the description of this parameter
        data_type (str): the data type for this parameter
    """
    _component_type = 'parameters'
    _builder = ParameterBuilder()

    name = ''
    description = ''
    data_type = 'mot_float_type'


class ProtocolParameterTemplate(ParameterTemplate):
    """The default template options for protocol parameters.

    To save memory, protocol data is loaded as a float by default.
    """
    data_type = 'float'
    value = None


class FreeParameterTemplate(ParameterTemplate):
    """The default template options for free parameters.

    Attributes:
        init_value (float): the initial value
        fixed (boolean or ndarray of float): if this parameter is fixed or not. If not fixed this should
            hold a reference to a value or a matrix
        lower_bound (float): the lower bounds, used in the parameter transform and prior
        upper_bound (float): the upper bounds, used in the parameter transform and prior
        parameter_transform
            (str or :class:`~mdt.model_building.parameter_functions.transformations.AbstractTransformation`): the
            parameter transformation, this is used for automatic range transformation of the parameters during
            optimization. See Harms 2017 NeuroImage for details. Typical elements are:

            * ``Identity``: no transformation
            * ``Positivity``: ensures the parameters are positive
            * ``Clamp``: limits the parameter between its lower and upper bounds
            * ``CosSqrClamp``: changes the range of the optimized parameters to [0, 1] and ensures boundary constraints
            * ``SinSqrClamp``: same as ``CosSqrClamp``
            * ``SqrClamp``: same as clamp but with an additional square root to change the magnitude of the range
            * ``AbsModPi``: ensures absolute modulus of the input parameters between zero and pi.
            * ``AbsModTwoPi``: ensures absolute modulus of the input parameters between zero and two pi.

        sampling_proposal_std (float): the default proposal standard deviation for this parameter. This is used
            in some MCMC sample routines.
        sampling_proposal_modulus (float or None): if given, a modulus we will use when finalizing the proposal
            continuous_distributions. That is, when we are finalizing the proposals we will take, if set, the absolute
            modulus of that parameter to ensure the parameter is within [0, <modulus>].
        sampling_prior: the prior function
        numdiff_info (dict or :class:`~mdt.model_building.parameter_functions.numdiff_info.NumDiffInfo`):
            the information necessary to take the numerical derivative of a model with respect to this parameter.
            Either a dictionary with the keyword arguments to
            :class:`~mdt.model_building.parameter_functions.numdiff_info.SimpleNumDiffInfo` or an information
            object directly. If None, we use an empty dictionary. Please note that if you override this, you will have
            to specify all of the items (no automatic inheritance of sub-items).
    """
    data_type = 'mot_float_type'
    fixed = False
    init_value = 1
    lower_bound = -1e20
    upper_bound = 1e20
    parameter_transform = 'Identity'
    sampling_proposal_std = 1
    sampling_proposal_modulus = None
    sampling_prior = UniformWithinBoundsPrior()
    numdiff_info = {'max_step': 0.1, 'scale_factor': 1, 'use_bounds': True,
                    'use_upper_bound': True, 'use_lower_bound': True}


def _resolve_parameter_transform(parameter_transform):
    """Resolves input parameter transforms to actual objects.

    Args:
        parameter_transform
            (str or :class:`~mdt.model_building.parameter_functions.transformations.AbstractTransformation`):
            a parameter transformation name (with or without the postfix ``Transform``) or an actual object we
            just return.

    Returns:
        mdt.model_building.parameter_functions.transformations.AbstractTransformation: an actual transformation object

    Raises:
        ValueError: if the parameter transformation could not be resolved.
    """
    if isinstance(parameter_transform, AbstractTransformation):
        return parameter_transform

    if has_component('parameter_transforms', parameter_transform):
        return get_component('parameter_transforms', parameter_transform)()

    if has_component('parameter_transforms', parameter_transform + 'Transform'):
        return get_component('parameter_transforms', parameter_transform + 'Transform')()

    raise ValueError('Could not resolve the parameter transformation "{}"'.format(parameter_transform))
