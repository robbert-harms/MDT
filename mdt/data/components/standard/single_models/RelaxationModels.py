from mdt.models.single import DMRISingleModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class S0TM(DMRISingleModelConfig):

    name = 'S0-TM'
    description = 'Model for the Mixing time.'
    model_expression = 'S0 * ExpT1DecTM'


class S0T2(DMRISingleModelConfig):

        name = 'S0-T2'
        description = 'Models the unweighted signal (aka. b0) with an extra T2.'
        model_expression = 'S0 * ExpT2Dec'


class S0T2T2(DMRISingleModelConfig):

        name = 'S0-T2T2'
        description = 'Model for the unweighted signal with two T2 models, one for short T2 and one for long T2.'

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


class S0_TEFA(DMRISingleModelConfig):

    name = 'S0-TEFA'
    description = 'Model for GRE data with variable TE and flip angle.'
    model_expression = 'S0 * ExpT1ExpT2GRE'
