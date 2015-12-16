from mdt.models.compartments import CLCodeFromInlineString, CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Ball(CompartmentConfig):

    name = 'Ball'
    cl_function_name = 'cmBall'
    parameter_list = ('b', 'd')
    cl_code = CLCodeFromInlineString('return exp(-d * b);')
