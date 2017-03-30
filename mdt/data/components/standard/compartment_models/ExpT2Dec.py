from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT2Dec(CompartmentConfig):

    parameter_list = ('TE', 'T2')
    cl_code = 'return exp(-TE / T2);'
