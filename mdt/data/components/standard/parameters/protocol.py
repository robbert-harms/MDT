"""Definitions of the protocol parameters

The type of these parameters signifies that the data for this parameter should come from the protocol defined in the
model data. These will never be optimized and are always set to the data defined in the protocol.

Please choose the parameter type for a model and parameter carefully since the type signifies how the parameter and
its data are handled during model construction.

"""
from mdt.component_templates.parameters import ProtocolParameterTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class g(ProtocolParameterTemplate):
    data_type = 'mot_float_type4'


class gx(ProtocolParameterTemplate):
    pass


class gy(ProtocolParameterTemplate):
    pass


class gz(ProtocolParameterTemplate):
    pass


class b(ProtocolParameterTemplate):
    pass


class G(ProtocolParameterTemplate):
    pass


class Delta(ProtocolParameterTemplate):
    pass


class delta(ProtocolParameterTemplate):
    pass


class TE(ProtocolParameterTemplate):
    pass


class TM(ProtocolParameterTemplate):
    pass


class TI(ProtocolParameterTemplate):
    pass


class TR(ProtocolParameterTemplate):
    pass


class flip_angle(ProtocolParameterTemplate):
    pass


# For STEAM/TSE sequences, depending on the model in which they are used.
class Refoc_fa1(ProtocolParameterTemplate):
    pass


# For STEAM/TSE sequences, depending on the model in which they are used.
class Refoc_fa2(ProtocolParameterTemplate):
    pass


# For STEAM/TSE sequences, depending on the model in which they are used.
class SEf(ProtocolParameterTemplate):
    pass
