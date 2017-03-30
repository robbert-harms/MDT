from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTR(CompartmentConfig):

    parameter_list = ('TR', 'T1')
    cl_code = 'return abs(1 - exp(-TR / T1));'
