import numpy as np
from mdt.component_templates.base import ComponentBuilder, ComponentTemplate
from mdt.lib.components import has_component, get_component
from mdt.model_building.parameter_functions.numdiff_info import NumDiffInfo, SimpleNumDiffInfo
from mdt.model_building.parameters import ProtocolParameter, FreeParameter, SphericalCoordinateParameter, \
    PolarAngleParameter, AzimuthAngleParameter, RotationalAngleParameter
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

            kwargs = dict(
                parameter_transform=_resolve_parameter_transform(template.parameter_transform),
                sampling_proposal_std=template.sampling_proposal_std,
                sampling_prior=template.sampling_prior,
                numdiff_info=numdiff_info
            )

            base_class = FreeParameter
            if issubclass(template, SphericalCoordinateParameterTemplate):
                base_class = SphericalCoordinateParameter
                if issubclass(template, PolarAngleParameterTemplate):
                    base_class = PolarAngleParameter
                if issubclass(template, AzimuthAngleParameterTemplate):
                    base_class = AzimuthAngleParameter
            elif issubclass(template, RotationalAngleParameterTemplate):
                base_class = RotationalAngleParameter
                kwargs.update(modulus=template.modulus)

            class AutoFreeParameter(base_class):
                def __init__(self, nickname=None):
                    super().__init__(
                        '{} {}'.format(template.data_type, nickname or template.name),
                        template.fixed,
                        template.init_value,
                        template.lower_bound,
                        template.upper_bound,
                        **kwargs
                    )

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
    sampling_prior = UniformWithinBoundsPrior()
    numdiff_info = {'max_step': 0.1, 'scale_factor': 1, 'use_bounds': True,
                    'use_upper_bound': True, 'use_lower_bound': True}


class SphericalCoordinateParameterTemplate(FreeParameterTemplate):
    """Template base class for spherical coordinate parameters.

    These are meant to be inherited by the polar angle template and the azimuth angle template.
    """
    init_value = np.pi / 2.0
    sampling_proposal_std = 0.1
    numdiff_info = {'max_step': 0.1, 'scale_factor': 10}


class PolarAngleParameterTemplate(SphericalCoordinateParameterTemplate):
    """Polar angle for use in spherical coordinate systems.

    If a compartment uses both a :class:`PolarAngleParameterTemplate` and :class:`AzimuthAngleParameterTemplate`,
    the composite model will ensure that the resulting cartesian coordinates are within the right spherical hemisphere.
    This is possible since diffusion is symmetric.

    In the background, we limit both the the polar angle and the azimuth angle between [0, pi] parameter
    between [0, pi] by projecting any other angle combination onto the right spherical hemisphere.
    """


class AzimuthAngleParameterTemplate(SphericalCoordinateParameterTemplate):
    """Azimuth angle for use in spherical coordinate systems.

    If a compartment uses both a :class:`PolarAngleParameterTemplate` and :class:`AzimuthAngleParameterTemplate`,
    the composite model will ensure that the resulting cartesian coordinates are within the right spherical hemisphere.
    This is possible since diffusion is symmetric.

    In the background, we limit both the the polar angle and the azimuth angle between [0, pi] parameter
    between [0, pi] by projecting any other angle combination onto the right spherical hemisphere.
    """


class RotationalAngleParameterTemplate(FreeParameterTemplate):
    """Template base class for parameters for which we want to enforce a modulus range.

    Parameters of this type are essentially unbounded, but their range is restricted to [0, modulus] using a modulo
    transformation. The modulus can be provided as an argument. This parameter class is recognized by the
    composite model which adds the necessary functions to the optimization and sampling routines.
    """
    init_value = np.pi / 2.0
    modulus = np.pi
    sampling_proposal_std = 0.1
    numdiff_info = {'max_step': 0.1, 'scale_factor': 10}


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
