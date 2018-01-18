from mdt.component_templates.composite_models import DMRICompositeModelTemplate

__author__ = 'Francisco.Lagos'


class S0_T1_GRE(DMRICompositeModelTemplate):

    description = '''Inversion recovery model. 
    
    This is made to model the MI-EPI sequence, a multi inversion recovery epi (Renvall et Al. 2016). 
    The Model is based on Stikov et al.'s three parameter model.
    '''
    model_expression = 'S0 * ExpT1DecIR'
    inits = {'ExpT1DecIR.T1': 3.0}
    upper_bounds = {'ExpT1DecIR.T1': 6.0}


class S0T2_exvivo(DMRICompositeModelTemplate):

    name = 'S0-T2-ExVivo'
    description = 'Models the unweighted signal (aka. b0) with an extra T2.'
    model_expression = 'S0 * ExpT2Dec'
    upper_bounds = {'ExpT2Dec.T2': 0.5}


class S0T2(DMRICompositeModelTemplate):

    name = 'S0-T2'
    description = 'Models the unweighted signal (aka. b0) with an extra T2.'
    model_expression = 'S0 * ExpT2Dec'
    upper_bounds = {'ExpT2Dec.T2': 2}
