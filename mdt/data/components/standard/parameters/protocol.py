"""Definitions of the protocol parameters.

The type of these parameters signifies that the data for this parameter should come from the protocol file or from
the protocol maps. These parameters are never optimized and are always set to the given input data.
"""
from mdt import ProtocolParameterTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class g(ProtocolParameterTemplate):
    data_type = 'float4'


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
    value = 1


class Refoc_fa1(ProtocolParameterTemplate):
    """For STEAM/TSE sequences, depending on the model in which they are used."""
    pass


class Refoc_fa2(ProtocolParameterTemplate):
    """For STEAM/TSE sequences, depending on the model in which they are used."""
    pass


class SEf(ProtocolParameterTemplate):
    """For STEAM/TSE sequences, depending on the model in which they are used."""
    pass


class b1(ProtocolParameterTemplate):
    value = 1


class Sw(ProtocolParameterTemplate):
    """This parameter is created only for linear T1 decay fitting of GRE data with variable flip angle.

    S_weighted is defined as the input data divided by the
        :math:`tan(flip_angle) ->  S_weighted = data / tan (flip_angle * B1_map)`
    """
    value = 1


class Dt(ProtocolParameterTemplate):
    """For STEAM/TSE sequences, depending on the model in which they are used."""
    pass


class excitation_b1(ProtocolParameterTemplate):
    """For STEAM/TSE sequences, depending on the model in which they are used."""
    pass


class refocusing1_b1(ProtocolParameterTemplate):
    """For STEAM/TSE sequences, depending on the model in which they are used."""
    pass


class refocusing2_b1(ProtocolParameterTemplate):
    """For STEAM/TSE sequences, depending on the model in which they are used."""
    pass
