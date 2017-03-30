from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco.Lagos'

# From protocol, if the signal is SE, we can setup TM = 0 in all the volumes,
# which returns to the standard SE signal decay


class ExpT1ExpT2STEAM(CompartmentConfig):
    """Generalised STEAM equation.

    From protocol, if the signal is SE, we can setup TM = 0 in all the volumes,
    which returns to the standard SE signal decay

    This equation can be used to calculate relaxation time (T1/T2) from spin echo (SE) and/or stimulated spin echo (STE)
    data. It is important to notice that in the protocol you have to define some parameters in a specific way:

    (1) For SE data, the original equation contains only the first refocusing pulse variable, but half of this value
        and in the power of two (sin(Refoc_fa1/2)**2). For that it is needed to define Refoc_fa2 = Refoc_fa1 and
        Refoc_fa1 has to be HALF of the used FA in the protocol (then, also Refoc_fa2). Also, the 0.5 factor is not included,
        then SEf (Spin echo flag) should be 0. Finally, TM (mixing time) has to be 0.
    (2) For STE data, this equation is used totally. Just SEf = 1.

    //UPDATE (24.03.17): This sequence is valid for STE SIGNAL ONLY! Don't mix with SE volumes.
    """

    parameter_list = ('SEf', 'TR', 'TE', 'TM', 'b', 'flip_angle', 'excitation_b1_map', 'Refoc_fa1', 'refocusing1_b1_map', 'Refoc_fa2', 'refocusing2_b1_map', 'T2', 'T1', 'd_exvivo')
    cl_code = """
        return sin((double)flip_angle * (double)excitation_b1_map)
            *   sin((double)Refoc_fa1 * (double)refocusing1_b1_map)
            *   sin((double)Refoc_fa2 * (double)refocusing2_b1_map)
            *   (exp(- (double)TM / (double) T1) - exp(- (double)TR / (double)T1))
            *   exp(- (double)TE / (double)T2)
            *   exp(- (double)b * (double)d_exvivo);
    """
