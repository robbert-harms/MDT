from mdt.models.compartments import CompartmentConfig, CLCodeFromInlineString

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Dot(CompartmentConfig):

    name = 'Dot'
    cl_function_name = 'cmDot'
    parameter_list = ()
    cl_code = CLCodeFromInlineString('return (MOT_FLOAT_TYPE)1.0;')
