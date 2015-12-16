from mdt.models.compartments import CompartmentConfig, CLCodeFromInlineString

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTR(CompartmentConfig):

    name = 'T1_IR'
    cl_function_name = 'cmExpT1DecTR'
    parameter_list = ('TR', 'T1')
    cl_code = CLCodeFromInlineString('return abs(1 - exp(-TR / T1));')
