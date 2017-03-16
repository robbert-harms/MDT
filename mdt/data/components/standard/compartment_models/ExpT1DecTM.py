from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco J. Fritz'
__date__ = "2016-09-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTM(CompartmentConfig):

    parameter_list = ('SEf', 'TR', 'TM', 'flip_angle', 'Refoc_fa1', 'Refoc_fa2', 'b', 'T1', 'd_exvivo')
    cl_code = """
        return powr((double)0.5, (double)SEf)
            * sin((double)flip_angle)
            * sin((double)Refoc_fa1)
            * sin((double)Refoc_fa2)
            * (1 - exp(-(TR - TM) / (double)T1))
            * exp(- ((TM * SEf) / (double)T1) - (double)(b * d_exvivo));
    """
