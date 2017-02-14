from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco J. Fritz'
__date__ = "2016-09-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTM(CompartmentConfig):

    parameter_list = ('SEf', 'flip_angle', 'Refoc_fa1', 'Refoc_fa2', 'TM', 'b', 'T1', 'b1_static', 'Dt_static')
    cl_code = 'return pow(0.5, SEf) ' \
              '* sin(flip_angle ' \
              '* b1_static) ' \
              '* sin(Refoc_fa1 * b1_static) ' \
              '* sin(Refoc_fa2 * b1_static) ' \
              '* exp(-TM / T1) ' \
              '* exp(-b * Dt_static);'
