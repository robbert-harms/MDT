from mdt import CompositeModelTemplate

__author__ = 'Francisco.Lagos'


class S0T1_MI_EPI(CompositeModelTemplate):
    """Inversion recovery model.
    
    This is made to model the MI-EPI sequence, a multi inversion recovery EPI (Renvall et Al. 2016). 
    The Model is based on Stikov et al.'s three parameter model.
    """
    model_expression = 'S0 * ExpT1DecIR'
    inits = {'ExpT1DecIR.T1': 3.0}
    upper_bounds = {'ExpT1DecIR.T1': 6.0}


class S0T1GRE(CompositeModelTemplate):
    """Gradient echo model.
        
    Models the unweighted signal (aka. b0) with an extra T1.
    
    This is the classical GRE T1-weighted equation, in which the signal is dependent of TR and flip angle.
    """
    name = 'S0-T1GRE'
    model_expression = 'S0 * ExpT1DecGRE'


class S0T2_exvivo(CompositeModelTemplate):
    """Models the unweighted signal (aka. b0) with an extra T2."""
    model_expression = 'S0 * ExpT2Dec'
    upper_bounds = {'ExpT2Dec.T2': 0.5}


class S0T2(CompositeModelTemplate):
    """Models the unweighted signal (aka. b0) with an extra T2."""
    model_expression = 'S0 * ExpT2Dec'


class MPM(CompositeModelTemplate):
    """Model for estimating biological microstructure of the tissue/sample."""
    model_expression = 'S0 * MPM_Fit'
    likelihood_function = 'Gaussian'
