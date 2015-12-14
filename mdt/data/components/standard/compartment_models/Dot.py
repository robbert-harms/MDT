from mdt.models.compartments import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Dot(DMRICompartmentModelBuilder):

    config = dict(
        name='Dot',
        cl_function_name='cmDot',
        parameter_list=(),
        cl_code_inline='return (MOT_FLOAT_TYPE)1.0;'
    )
