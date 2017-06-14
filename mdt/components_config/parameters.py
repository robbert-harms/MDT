import six
from mdt.components_loader import ComponentConfig, ComponentBuilder, method_binding_meta
from mot.cl_data_type import SimpleCLDataType
from mot.model_building.parameters import StaticMapParameter, ProtocolParameter, ModelDataParameter, \
    FreeParameter
from mot.model_building.parameter_functions.priors import UniformWithinBoundsPrior
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mot.model_building.parameter_functions.sample_statistics import GaussianPSS
from mot.model_building.parameter_functions.transformations import IdentityTransform

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterConfig(ComponentConfig):
    """The cascade config to inherit from.

    These configs are loaded on the fly by the ParametersBuilder

    Config options:
        name (str): the name of the parameter, defaults to the class name
        description (str): the description of this parameter
        data_type (str or DataType): either a string we use as datatype or the actual datatype itself
        type (str): the type of parameter (free, protocol or model_data)
    """
    name = ''
    description = ''
    data_type = 'mot_float_type'
    type = None


class ProtocolParameterConfig(ParameterConfig):
    """The default config options for protocol parameters.

    This sets the attribute type to protocol.
    """
    type = 'protocol'
    data_type = 'mot_float_type'


class FreeParameterConfig(ParameterConfig):
    """The default config options for free parameters.

    This sets the attribute type to free.

    Attributes:
        init_value (float): the initial value
        fixed (boolean or ndarray of float): if this parameter is fixed or not. If not fixed this should
            hold a reference to a value or a matrix
        lower_bound (float): the lower bounds
        upper_bound (float): the upper bounds
        parameter_transform: the parameter transformation
        sampling_proposal: the proposal function
        sampling_prior: the prior function
        sampling_statistics: the sampling statistic, used after the sampling
    """
    type = 'free'
    data_type = 'mot_float_type'
    fixed = False
    init_value = 0.03
    lower_bound = 0.0
    upper_bound = 4.0
    parameter_transform = IdentityTransform()
    sampling_proposal = GaussianProposal(1.0)
    sampling_prior = UniformWithinBoundsPrior()
    sampling_statistics = GaussianPSS()


class ModelDataParameterConfig(ParameterConfig):
    """The default config options for model data parameters.

    This sets the attribute type to model_data.
    """
    type = 'model_data'
    value = None


class StaticMapParameterConfig(ParameterConfig):
    """The default config options for static data parameters.

    This sets the attribute type to static_map.
    """
    type = 'static_map'
    value = None


class ParameterBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class DMRICompositeModel

        Args:
            template (ParameterConfig): the configuration for the parameter.
        """
        data_type = template.data_type
        if isinstance(data_type, six.string_types):
            data_type = SimpleCLDataType.from_string(data_type)

        if template.type.lower() == 'protocol':
            class AutoProtocolParameter(method_binding_meta(template, ProtocolParameter)):
                def __init__(self):
                    super(AutoProtocolParameter, self).__init__(data_type, template.name)
            return AutoProtocolParameter

        elif template.type.lower() == 'free':
            class AutoFreeParameter(method_binding_meta(template, FreeParameter)):
                def __init__(self):
                    super(AutoFreeParameter, self).__init__(
                        data_type,
                        template.name,
                        template.fixed,
                        template.init_value,
                        template.lower_bound,
                        template.upper_bound,
                        parameter_transform=template.parameter_transform,
                        sampling_proposal=template.sampling_proposal,
                        sampling_prior=template.sampling_prior,
                        sampling_statistics=template.sampling_statistics
                    )
            return AutoFreeParameter

        elif template.type.lower() == 'model_data':
            class AutoModelDataParameter(method_binding_meta(template, ModelDataParameter)):
                def __init__(self):
                    super(AutoModelDataParameter, self).__init__(data_type, template.name, template.value)
            return AutoModelDataParameter

        elif template.type.lower() == 'static_map':
            class AutoStaticMapParameter(method_binding_meta(template, StaticMapParameter)):
                def __init__(self):
                    super(AutoStaticMapParameter, self).__init__(data_type, template.name, template.value)
            return AutoStaticMapParameter
