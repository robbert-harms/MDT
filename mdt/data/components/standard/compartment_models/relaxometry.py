from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Francisco.Lagos'
__licence__ = 'LGPL v3'


class ExpT1DecGRE(CompartmentTemplate):

    parameter_list = ('TR', 'flip_angle', 'excitation_b1_map', 'T1')
    cl_code = """
        return sin((double)flip_angle * excitation_b1_map) *
              (1 - exp(-TR / (double)T1)) /
              (1 - cos((double)flip_angle * excitation_b1_map) * exp(-TR / (double)T1) );
    """


class ExpT1DecIR(CompartmentTemplate):
    description = """IR equation.

    IR equation in which TI and TR are considered to estimate T1. Assuming TE << T1, the TE component of the signal
    is discarded. In cascade, S0 contains T2 and PD weighted information. An efficiency factor is added to the TI
    parameter.
    
    This is made to model the MI-EPI sequence, a multi inversion recovery epi (Renvall et Al. 2016). 
    The Model is based on Stikov et al.'s three parameter model.
    """
    parameter_list = ('TR', 'TI', 'Efficiency', 'T1')
    cl_code = """
        return fabs(1 + exp(-TR / (double)T1) - 2 * Efficiency * exp(-TI / (double)T1));
    """


class ExpT1DecTM(CompartmentTemplate):

    parameter_list = ('SEf', 'TR', 'TM', 'TE', 'flip_angle', 'excitation_b1_map', 'Refoc_fa1', 'refocusing1_b1_map',
                      'Refoc_fa2', 'refocusing2_b1_map', 'b', 'T1', 'd_exvivo')
    cl_code = """
        return powr((double)0.5, (double)SEf)
            * sin((double)flip_angle * excitation_b1_map)
            * sin((double)Refoc_fa1 * refocusing1_b1_map)
            * sin((double)Refoc_fa2 * (refocusing2_b1_map * SEf + refocusing1_b1_map * (1 - SEf)))
            * (1 - exp(-(TR - TM) / (double)T1))
            * exp(- ((TM * SEf) / (double)T1) - (double)(b * d_exvivo));
    """


class ExpT1DecTM_simple(CompartmentTemplate):

    parameter_list = ('TM', 'T1')
    cl_code = 'return exp(-TM / T1);'


class ExpT1DecTR(CompartmentTemplate):

    parameter_list = ('TR', 'T1')
    cl_code = 'return abs(1 - exp(-TR / T1));'


class ExpT1ExpT2GRE(CompartmentTemplate):

    parameter_list = ('TR', 'TE', 'flip_angle', 'T1', 'T2')
    cl_code = """
        return sin(flip_angle) * (1 - exp(-TR / T1)) / (1 - cos(flip_angle) * exp(-TR / T1)) * exp(-TE / T2);
    """


class ExpT1ExpT2sGRE(CompartmentTemplate):

    parameter_list = ('TR', 'TE', 'flip_angle', 'T1', 'T2s')
    cl_code = """
        return sin(flip_angle) * (1 - exp(-TR / T1)) / (1 - cos(flip_angle) * exp(-TR / T1)) * exp(-TE / T2s);
    """


class ExpT1ExpT2STEAM(CompartmentTemplate):
    description = """Generalised STEAM equation.

    From protocol, if the signal is SE, we can setup TM = 0 in all the volumes,
    which returns to the standard SE signal decay

    This equation can be used to calculate relaxation time (T1/T2) from spin echo (SE) and/or stimulated spin echo (STE)
    data. It is important to notice that in the protocol you have to define some parameters in a specific way:

    (1) For SE data, the original equation contains only the first refocusing pulse variable, but half of this value
        and in the power of two (sin(Refoc_fa1/2)**2). For that it is needed to define Refoc_fa2 = Refoc_fa1 and
        Refoc_fa1 has to be HALF of the used FA in the protocol (then, also Refoc_fa2). Also, the 0.5 factor is
        not included, then SEf (Spin echo flag) should be 0. Finally, TM (mixing time) has to be 0.
    (2) For STE data, this equation is used totally. Just SEf = 1.

    //UPDATE (24.03.17): This sequence is valid for STE SIGNAL ONLY! Don't mix with SE volumes.
    """
    parameter_list = ('SEf', 'TR', 'TE', 'TM', 'b', 'flip_angle', 'excitation_b1_map', 'Refoc_fa1',
                      'refocusing1_b1_map', 'Refoc_fa2', 'refocusing2_b1_map', 'T2', 'T1', 'd_exvivo')
    cl_code = """
        return sin((double)flip_angle * (double)excitation_b1_map)
            *   sin((double)Refoc_fa1 * (double)refocusing1_b1_map)
            *   sin((double)Refoc_fa2 * (double)refocusing2_b1_map)
            *   (exp(- (double)TM / (double) T1) - exp(- (double)TR / (double)T1))
            *   exp(- (double)TE / (double)T2)
            *   exp(- (double)b * (double)d_exvivo);
    """


class ExpT2Dec(CompartmentTemplate):

    parameter_list = ('TE', 'T2')
    cl_code = 'return exp(-TE / T2);'


class ExpT2DecSTEAM(CompartmentTemplate):

    parameter_list = ('SEf', 'TE', 'TM', 'b', 'flip_angle', 'excitation_b1_map', 'Refoc_fa1',
                      'refocusing1_b1_map', 'Refoc_fa2', 'refocusing2_b1_map', 'T2', 'T1', 'd_exvivo')
    cl_code = """
        return powr((double)0.5, (double)SEf)
            *   sin((double)flip_angle * (double)excitation_b1_map)
            *   sin((double)Refoc_fa1 * (double)refocusing1_b1_map)
            *   sin((double)Refoc_fa2 * ((double)refocusing2_b1_map * (double)SEf + 
                                         (double)refocusing1_b1_map * (double)(1 - SEf)))
            *   exp(- (double)TE / (double)T2)
            *   exp(- (double)TM * SEf / (double) T1)
            *   exp(- (double)b * (double)d_exvivo);
    """


class MPM_Fit(CompartmentTemplate):
    description = """MPM fitting (Weiskopf, 2016 ESMRMB Workshop)

    This fitting is a model published by Helms (2008) and Weiskopf (2011) to determinate biological properties
    of the tissue/sample in function *of several images*, which includes T1w, PDw and MTw images. 
    This function is still an approximation and only if the assumptions of the approximations hold for ex-vivo tissue, 
    then can be used for ex-vivo data.
    """
    parameter_list = ('TR', 'flip_angle', 'excitation_b1_map', 'T1')
    cl_code = 'return (flip_angle * excitation_b1_map) * ( (TR / T1) ' \
              '     / ( pown(flip_angle * excitation_b1_map, 2) / 2 + ( TR / T1 ) ) );'


class LinMPM_Fit(CompartmentTemplate):
    description = """MPM fitting (Weiskopf, 2016 ESMRMB Workshop)

    This fitting is a model published by Helms (2008) and Weiskopf (2011) to determinate biological properties
    of the tissue/sample in function *of several images*, which includes T1w, PDw and MTw images. 
    This function is still an approximation and only if the assumptions of the approximations hold for ex-vivo tissue, 
    then can be used for ex-vivo data.
    """
    parameter_list = ('TR', 'flip_angle', 'b1_static', 'T1')
    cl_code = 'return log(flip_angle * b1_static) + log(TR / T1)' \
              '       - log( pown(flip_angle * b1_static, 2) / 2 + ( TR / T1 ) ) ;'


class LinT1GRE(CompartmentTemplate):
    description = """Lineal T1 fitting (Weiskopf, 2016 ESMRMB Workshop)

    This fitting is the extension of the standard GRE equation for flip angles lower than 90deg. This modelling allows a
    linear fitting of the data if is enough data to support it. In principle, it should not be a problem if only two
    points are used, however the addition of a constant in the equation could give some kind of uncertainty.

    B1 has to be normalized *in function of the reference voltage, the angle distribution and the reference angle*.
    Here I assume that TR <<< T1, then exp(-TR/T1) ~ 1 - TR/T1. Then the equation becomes 'simpler'. 
    However, if this condition is not achieved, then return to the standard equation.
    
    Also, DATA HAS TO BE PROCESSED BEFORE TO USE THIS EQUATION. Please apply log() on the data.
    """
    parameter_list = ('Sw_static', 'E1')
    cl_code = """
        return Sw_static * E1;
    """


class LinT2Dec(CompartmentTemplate):

    parameter_list = ('TE', 'R2')
    cl_code = 'return -TE * R2;'

