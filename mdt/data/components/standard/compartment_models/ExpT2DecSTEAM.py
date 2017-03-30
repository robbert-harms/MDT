from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco J. Fritz'
__date__ = "2016-09-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT2DecSTEAM(CompartmentConfig):

    parameter_list = ('SEf', 'TE', 'TM', 'b', 'flip_angle', 'excitation_b1_map', 'Refoc_fa1', 'refocusing1_b1_map', 'Refoc_fa2', 'refocusing2_b1_map', 'T2', 'T1', 'd_exvivo')
    cl_code = """
        return powr((double)0.5, (double)SEf)
            *   sin((double)flip_angle * (double)excitation_b1_map)
            *   sin((double)Refoc_fa1 * (double)refocusing1_b1_map)
            *   sin((double)Refoc_fa2 * ((double)refocusing2_b1_map * (double)SEf + (double)refocusing1_b1_map * (double)(1 - SEf)))
            *   exp(- (double)TE / (double)T2)
            *   exp(- (double)TM * SEf / (double) T1)
            *   exp(- (double)b * (double)d_exvivo);
    """
