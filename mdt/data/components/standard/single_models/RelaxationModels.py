from mot.model_building.evaluation_models import GaussianEvaluationModel
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
        description = 'Models the unweighted signal (aka. b0) with an extra T2.'
        model_expression = 'S0 * ExpT2Dec'

        # for proper initialization, please take the highest S0 value in your data.
        inits = {'S0.s0': 50.0}
        upper_bounds = {'ExpT2Dec.T2': 0.10,
                        'S0.s0': 150}


class S0LinT2(DMRISingleModelConfig):

        name = 'S0LinT2'
        description = 'Models the unweighted signal (aka. b0) with an extra T2.'
        model_expression = 'S0 + LinT2Dec'
        upper_bounds = {'LinT2Dec.R2': -1000,
                        'S0.s0': 7.0}  # In this model, S0 is actually ln(S0).


class S0T2(DMRISingleModelConfig):

        name = 'S0-T2'
        description = 'Models the unweighted text_message_signal (aka. b0) with an extra T2.'
        model_expression = 'S0 * ExpT2Dec'

        # for proper initialization, please take the highest S0 value in your data.
        #inits = {'S0.s0': 50.0}
        upper_bounds = {'ExpT2Dec.T2': 0.15}


class S0_IRT1(DMRISingleModelConfig):

        name = 'S0-ExpT1DecIR'
        description = 'Model with multi-IR data (?)'
        model_expression = 'S0 * ExpT1DecIR'

        # for proper initialization, please take the highest S0 value in your data.
        inits = {'S0.s0': 50.0}
        upper_bounds = {'ExpT1DecIR.T2': 0.10,
                        'S0.s0': 150}


class S0T1GRE(DMRISingleModelConfig):

        name = 'S0-T1GRE'
        description = 'Models the unweighted text_message_signal (aka. b0) with an extra T1.'
        model_expression = 'S0 * ExpT1DecGRE'

        # for proper initialization, please take the highest S0 value in your data.
        inits = {'S0.s0': 50.0}
        upper_bounds = {'ExpT1DecGRE.T1': 1.0,
                        'S0.s0': 150}


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


class GRE_Relax(DMRISingleModelConfig):

    name = 'GRE_Relax'
    description = 'Model for estimating T1 and T2 from GRE data with variable TE, TR and flip angle.'
    model_expression = 'S0 * ExpT1ExpT2GRE'
    inits = {'ExpT1ExpT2GRE.T1': 0.2,
             'ExpT1ExpT2GRE.T2': 0.05,
             'S0.s0': 1e3}
    upper_bounds = {'ExpT1ExpT2GRE.T1': 1,
                    'ExpT1ExpT2GRE.T2': 0.5,
                    'S0.s0': 1e4}


class STEAM_Relax(DMRISingleModelConfig):

    name = 'STEAM_Relax'
    description = 'Model for estimating T1 and T2 from data with a variable TM and TE.'
    model_expression = 'S0 * ExpT1ExpT2STEAM'
    inits = {'ExpT1ExpT2STEAM.T2': 0.03,
             'ExpT1ExpT2STEAM.T1': 0.15}
    upper_bounds = {'ExpT1ExpT2STEAM.T1': 0.5,
                    'ExpT1ExpT2STEAM.T2': 0.1}


class S0LinearGRE(DMRISingleModelConfig):
    # S0 is not the "real" s0 of the data, it is s0*(1 - exp(-TR / T1)). Then, s0 can be only calculated AFTER T1 estimation.
    name = 'S0-LinearGRE'
    description = 'Model for estimating T1 of GRE data using B1+ map and several FA variations.'
    model_expression = 'S0 + LinT1GRE'
    evaluation_model = GaussianEvaluationModel()


class MPM(DMRISingleModelConfig):

    name = 'MPM'
    description = 'Model for estimating biological microstructure of the tissue/sample.'
    model_expression = 'S0 * MPM_Fit'
