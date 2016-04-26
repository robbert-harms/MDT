from mdt.models.compartments import CompartmentConfig, CLCodeFromInlineString

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTM(CompartmentConfig):

    name = 'ExpT1DecTM'
    cl_function_name = 'cmExpT1DecTM'
    parameter_list = ('TM', 'T1')
    cl_code = CLCodeFromInlineString('return exp( -TM / T1 );')
