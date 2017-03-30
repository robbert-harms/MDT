from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco J. Fritz'
__date__ = "2016-09-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTM(CompartmentConfig):

    parameter_list = ('SEf', 'TR', 'TM', 'TE', 'flip_angle', 'excitation_b1_map', 'Refoc_fa1', 'refocusing1_b1_map', 'Refoc_fa2', 'refocusing2_b1_map', 'b', 'T1', 'd_exvivo')
    cl_code = """
        return powr((double)0.5, (double)SEf)
            * sin((double)flip_angle * excitation_b1_map)
            * sin((double)Refoc_fa1 * refocusing1_b1_map)
            * sin((double)Refoc_fa2 * (refocusing2_b1_map * SEf + refocusing1_b1_map * (1 - SEf)))
            * (1 - exp(-(TR - TM) / (double)T1))
            * exp(- ((TM * SEf) / (double)T1) - (double)(b * d_exvivo));
    """
