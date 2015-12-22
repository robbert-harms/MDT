from mdt.models.parameters import ProtocolParameterConfig

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class g(ProtocolParameterConfig):
    name = 'g'
    data_type = 'MOT_FLOAT_TYPE4'


class b(ProtocolParameterConfig):
    name = 'b'


class G(ProtocolParameterConfig):
    name = 'G'


class Delta(ProtocolParameterConfig):
    name = 'Delta'


class delta(ProtocolParameterConfig):
    name = 'delta'


class q(ProtocolParameterConfig):
    name = 'q'


class GGD(ProtocolParameterConfig):
    name = 'GAMMA2_G2_delta2'


class TE(ProtocolParameterConfig):
    name = 'TE'


class TM(ProtocolParameterConfig):
    name = 'TM'


class Ti(ProtocolParameterConfig):
    name = 'Ti'


class TR(ProtocolParameterConfig):
    name = 'TR'
