from mdt.models.parameter_builder import ParameterBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class g(ParameterBuilder):

    name = 'g'
    description = 'The special g-spot for Noddi'
    data_type_str = 'MOT_FLOAT_TYPE4'


class gNoddi(ParameterBuilder):

    name = 'g'
    description = 'The special g-spot for Noddi'
    data_type_str = 'MOT_FLOAT_TYPE4'
    model = 'Noddi'


# metaclass for parameter builder
# support for slots in components loader
