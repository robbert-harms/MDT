from mdt.models.single import DMRISingleModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class S0TM(DMRISingleModelConfig):

    name = 'S0-TM'
    description = 'Model for the Mixing time.'
    model_expression = 'S0 * ExpT1DecTM'
    #upper_bounds = {'T1.T1': 0.5}


class S0T2(DMRISingleModelConfig):

        name = 'S0-T2'
        description = 'Models the unweighted text_message_signal (aka. b0) with an extra T2.'
        model_expression = 'S0 * ExpT2Dec'
        #upper_bounds = {'T2': 0.08}


class S0T2T2(DMRISingleModelConfig):

        name = 'S0-T2T2'
        description = 'Model for the unweighted text_message_signal with two T2 models, one for short T2 and one for long T2.'

        model_expression = '''
            S0 * ( (Weight(w_long) * ExpT2Dec(T2_long)) +
                   (Weight(w_short) * ExpT2Dec(T2_short))
                 )
        '''

        fixes = {'T2_long.T2': 0.5}
        upper_bounds = {'T2_short.T2': 0.08}

        post_optimization_modifiers = (
            ('T2_short.T2Weighted', lambda d: d['w_short.w'] * d['T2_short.T2']),
            ('T2_long.T2Weighted', lambda d: d['w_long.w'] * d['T2_long.T2']),
            ('T2.T2', lambda d: d['T2_short.T2Weighted'] + d['T2_long.T2Weighted'])
        )


class GRE_Relax(DMRISingleModelConfig):

    name = 'GRE_Relax'
    description = 'Model for estimating T1 and T2 from GRE data with variable TE, TR and flip angle.'
    model_expression = 'S0 * ExpT1ExpT2GRE'
    inits = {'ExpT1ExpT2GRE.T1': 0.3,
             'ExpT1ExpT2GRE.T2': 0.05}
    upper_bounds = {'ExpT1ExpT2GRE.T1': 1,
                    'ExpT1ExpT2GRE.T2': 0.5}


class STEAM_Relax(DMRISingleModelConfig):

    name = 'STEAM_Relax'
    description = 'Model for estimating T1 and T2 from data with a variable TM and TE.'
    model_expression = 'S0 * ExpT1ExpT2STEAM'
    inits = {
             'ExpT1ExpT2STEAM.T2': 0.03}
    fixes = {'ExpT1ExpT2STEAM.T1': 0.125}
    upper_bounds = {'ExpT1ExpT2STEAM.T1': 1,
                    'ExpT1ExpT2STEAM.T2': 0.1}
