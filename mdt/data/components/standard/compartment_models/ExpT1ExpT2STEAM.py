from mdt.models.compartments import CompartmentConfig

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
    """

    parameter_list = ('SEf', 'TM', 'TE', 'flip_angle', 'Refoc_fa1', 'Refoc_fa2', 'T1', 'T2')
    cl_code = """
        return pow(0.5, SEf) * sin(flip_angle) * sin(Refoc_fa1) * sin(Refoc_fa2) * exp(-TE / T2) * exp(-TM / T1);
    """
