from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2017-07-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTM_simple(CompartmentTemplate):

    parameter_list = ('TM', 'T1')
    cl_code = 'return exp(-TM / T1);'
