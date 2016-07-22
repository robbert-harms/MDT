from mdt.models.compartments import CLCodeFromInlineString, CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class S0(CompartmentConfig):

    name = 'S0'
    cl_function_name = 'cmS0'
    parameter_list = ('s0', '_observation')
    cl_code = CLCodeFromInlineString('return s0 * _observation;')
