from mdt import CompartmentTemplate, FreeParameterTemplate

__author__ = 'Francisco Fritz'
__licence__ = 'LGPL v3'


class ExpT1DecGRE(CompartmentTemplate):

    parameters = ('TR', 'flip_angle', 'excitation_b1', 'T1')
    cl_code = '''
        return sin(flip_angle * excitation_b1) *
              (1 - exp(-TR / T1)) /
              (1 - cos(flip_angle * excitation_b1) * exp(-TR / T1) );
    '''


class ExpT1DecIR(CompartmentTemplate):
    """IR equation.

    IR equation in which TI and TR are considered to estimate T1. Assuming TE << T1, the TE component of the signal
    is discarded. In cascade, S0 contains T2 and PD weighted information. An efficiency factor is added to the TI
    parameter.
    
    This is made to model the MI-EPI sequence, a multi inversion recovery epi (Renvall et Al. 2016). 
    The Model is based on Stikov et al.'s three parameter model.
    """
    parameters = ('TR', 'TI', 'Efficiency', 'T1')
    cl_code = '''
        return fabs(1 + exp(-TR / T1) - 2 * Efficiency * exp(-TI / T1));
    '''

    class Efficiency(FreeParameterTemplate):
        init_value = 0.95
        lower_bound = 0
        upper_bound = 1
        parameter_transform = 'SinSqrClamp'
        sampling_proposal_std = 0.001


class ExpT1DecTM(CompartmentTemplate):

    parameters = ('SEf', 'TR', 'TM', 'TE', 'T1', 'flip_angle', 'excitation_b1', 'Refoc_fa1',
                  'refocusing1_b1', 'Refoc_fa2', 'refocusing2_b1', 'b', 'd_exvivo')
    cl_code = '''
        return powr((double)0.5, SEf)
            * sin(flip_angle * excitation_b1)
            * sin(Refoc_fa1 * refocusing1_b1)
            * sin(Refoc_fa2 * (refocusing2_b1 * SEf + refocusing1_b1 * (1 - SEf)))
            * (1 - exp(-(TR - TM) / T1))
            * exp(- ((TM * SEf) / T1) - (b * d_exvivo));
    '''


class ExpT1DecTM_simple(CompartmentTemplate):

    parameters = ('TM', 'T1')
    cl_code = 'return exp(-TM / T1);'


class ExpT1DecTR(CompartmentTemplate):

    parameters = ('TR', 'T1')
    cl_code = 'return abs(1 - exp(-TR / T1));'


class ExpT1ExpT2GRE(CompartmentTemplate):

    parameters = ('TR', 'TE', 'excitation_b1', 'flip_angle', 'T1', 'T2')
    cl_code = '''
        return sin(flip_angle * excitation_b1) * (1 - exp(-TR / T1)) 
                / (1 - cos(flip_angle * excitation_b1) * exp(-TR / T1)) * exp(-TE / T2);
    '''


class ExpT1ExpT2sGRE(CompartmentTemplate):

    parameters = ('TR', 'TE', 'excitation_b1', 'flip_angle', 'T1', 'T2s')
    cl_code = '''
        return sin(flip_angle * excitation_b1) * (1 - exp(-TR / T1)) 
                / (1 - cos(flip_angle * excitation_b1) * exp(-TR / T1)) * exp(-TE / T2s);
    '''


class ExpT1ExpT2STEAM(CompartmentTemplate):
    """Generalised STEAM equation.

    From protocol, if the signal is SE, we can setup TM = 0 in all the volumes,
    which returns to the standard SE signal decay

    Please note, this sequence is valid for STE SIGNAL ONLY! Don't mix with SE volumes.

    This equation can be used to calculate relaxation time (T1/T2) from spin echo (SE) and/or stimulated spin echo (STE)
    data. It is important to notice that in the protocol you have to define some parameters in a specific way:

    (1) For SE data, the original equation contains only the first refocusing pulse variable, but half of this value
        and in the power of two (sin(Refoc_fa1/2)**2). For that it is needed to define Refoc_fa2 = Refoc_fa1 and
        Refoc_fa1 has to be HALF of the used FA in the protocol (then, also Refoc_fa2). Also, the 0.5 factor is
        not included, then SEf (Spin echo flag) should be 0. Finally, TM (mixing time) has to be 0.
    (2) For STE data, this equation is used totally. Just SEf = 1.
    """
    parameters = ('SEf', 'TR', 'TE', 'TM', 'b', 'flip_angle', 'excitation_b1', 'Refoc_fa1',
                  'refocusing1_b1', 'Refoc_fa2', 'refocusing2_b1', 'T2', 'T1', 'd_exvivo')
    cl_code = '''
        return sin(flip_angle * excitation_b1)
            *   sin(Refoc_fa1 * refocusing1_b1)
            *   sin(Refoc_fa2 * refocusing2_b1)
            *   (exp(- TM /  T1) - exp(- TR / T1))
            *   exp(- TE / T2)
            *   exp(- b * d_exvivo);
    '''


class ExpT2Dec(CompartmentTemplate):

    parameters = ('TE', 'T2')
    cl_code = 'return exp(-TE / T2);'


class ExpT2DecSTEAM(CompartmentTemplate):

    parameters = ('SEf', 'TE', 'TM', 'b', 'flip_angle', 'excitation_b1', 'Refoc_fa1',
                      'refocusing1_b1', 'Refoc_fa2', 'refocusing2_b1', 'T2', 'T1', 'd_exvivo')
    cl_code = '''
        return powr((double)0.5, SEf)
            *   sin(flip_angle * excitation_b1)
            *   sin(Refoc_fa1 * refocusing1_b1)
            *   sin(Refoc_fa2 * (refocusing2_b1 * SEf + refocusing1_b1 * (1 - SEf)))
            *   exp(- TE / T2)
            *   exp(- TM * SEf /  T1)
            *   exp(- b * d_exvivo);
    '''


class MPM_Fit(CompartmentTemplate):
    """MPM fitting (Weiskopf, 2016 ESMRMB Workshop)

    This fitting is a model published by Helms (2008) and Weiskopf (2011) to determinate biological properties
    of the tissue/sample in function *of several images*, which includes T1w, PDw and MTw images. 
    This function is still an approximation and only if the assumptions of the approximations hold for ex-vivo tissue, 
    then can be used for ex-vivo data.
    """
    parameters = ('TR', 'flip_angle', 'excitation_b1', 'T1')
    cl_code = '''
        return (flip_angle * excitation_b1) * ( (TR / T1) / ( pown(flip_angle * excitation_b1, 2) / 2 + ( TR / T1 ) ) );
    '''


class LinMPM_Fit(CompartmentTemplate):
    """MPM fitting (Weiskopf, 2016 ESMRMB Workshop)

    This fitting is a model published by Helms (2008) and Weiskopf (2011) to determinate biological properties
    of the tissue/sample in function *of several images*, which includes T1w, PDw and MTw images. 
    This function is still an approximation and only if the assumptions of the approximations hold for ex-vivo tissue, 
    then can be used for ex-vivo data.
    """
    parameters = ('TR', 'flip_angle', 'b1', 'T1')
    cl_code = '''
        return log(flip_angle * b1) + log(TR / T1) - log( pown(flip_angle * b1, 2) / 2 + ( TR / T1 ) ) ;
    '''


class LinT1GRE(CompartmentTemplate):
    """Linear T1 fitting (Weiskopf, 2016 ESMRMB Workshop)

    This fitting is the extension of the standard GRE equation for flip angles lower than 90deg. This modelling allows a
    linear fitting of the data if is enough data to support it. In principle, it should not be a problem if only two
    points are used, however the addition of a constant in the equation could give some kind of uncertainty.

    B1 has to be normalized *in function of the reference voltage, the angle distribution and the reference angle*.
    Here I assume that TR <<< T1, then exp(-TR/T1) ~ 1 - TR/T1. Then the equation becomes 'simpler'. 
    However, if this condition is not achieved, then return to the standard equation.
    
    Also, DATA HAS TO BE PROCESSED BEFORE TO USE THIS EQUATION. Please apply log() on the data.
    """
    parameters = ('Sw', 'E1')
    cl_code = '''
        return Sw * E1;
    '''

    class E1(FreeParameterTemplate):
        """This parameter is defined *only* for linear decay T1 fitting in GRE data *with* TR constant.

        This parameter is also defined in the SSFP equation. However, in SSFP this parameter is from the protocol (!)
            E1 = exp( -TR / T1 ).
        After estimation of this parameter, T1 can be recovered by applying the next equation:
            -TR / log( E1 ).
        """
        init_value = 0.37
        lower_bound = 0.0
        upper_bound = 1.0


class LinT2Dec(CompartmentTemplate):

    parameters = ('TE', 'R2')
    cl_code = 'return -TE * R2;'

