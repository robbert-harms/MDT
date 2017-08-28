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
