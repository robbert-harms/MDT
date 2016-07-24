from mdt.models.compartments import CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecIR(CompartmentConfig):

    parameter_list = ('Ti', 'TR', 'E', 'T1')
    cl_code = 'return abs(1 - 2 * E * exp(-Ti / T1) + exp( - TR / T1) );'
