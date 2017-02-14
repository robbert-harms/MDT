"""Definitions of the protocol parameters

The type of these parameters signifies that the data for this parameter should come from the protocol defined in the
model data. These will never be optimized and are always set to the data defined in the protocol.

Please choose the parameter type for a model and parameter carefully since the type signifies how the parameter and
its data are handled during model construction.

"""
from mdt.components_config.parameters import ProtocolParameterConfig

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class g(ProtocolParameterConfig):
    data_type = 'mot_float_type4'


class b(ProtocolParameterConfig):
    pass


class G(ProtocolParameterConfig):
    pass


class Delta(ProtocolParameterConfig):
    pass


class delta(ProtocolParameterConfig):
    pass


class TE(ProtocolParameterConfig):
    pass


class TM(ProtocolParameterConfig):
    pass


class Ti(ProtocolParameterConfig):
    pass


class TR(ProtocolParameterConfig):
    pass


class flip_angle(ProtocolParameterConfig):
    pass

# For STEAM/TSE sequences, depending on the model in which they are used.
class Refoc_fa1(ProtocolParameterConfig):
    pass


# For STEAM/TSE sequences, depending on the model in which they are used.
class Refoc_fa2(ProtocolParameterConfig):
    pass


# For STEAM/TSE sequences, depending on the model in which they are used.
class SEf(ProtocolParameterConfig):
    pass
