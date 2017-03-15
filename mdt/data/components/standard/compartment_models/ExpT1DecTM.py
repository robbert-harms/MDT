from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco J. Fritz'
__date__ = "2016-09-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTM(CompartmentConfig):

    parameter_list = ('SEf', 'TR', 'TM', 'flip_angle', 'Refoc_fa1', 'Refoc_fa2', 'T1', 'b', 'd_exvivo')
    cl_code = """
        return pow(0.5, SEf) * (1 - exp(-(TR - TM) / T1)) * sin(flip_angle) * sin(Refoc_fa1) * sin(Refoc_fa2) * exp(- (TM * SEf) / T1) * exp( - b * d_exvivo);
    """
