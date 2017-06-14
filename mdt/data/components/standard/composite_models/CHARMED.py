from mdt.components_config.composite_models import DMRICompositeModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CHARMED_r1(DMRICompositeModelConfig):

    description = 'The CHARMED model with 1 restricted compartments'

    model_expression = '''
        S0 * ( (Weight(w_hin0) * Tensor) +
               (Weight(w_res0) * CHARMEDRestricted(CHARMEDRestricted0))
               )
    '''

    lower_bounds = {'Tensor.d': 1e-9,
                    'Tensor.dperp0': 0.3e-9,
                    'Tensor.dperp1': 0.3e-9,
                    'CHARMEDRestricted0.d': 0.3e-9}

    upper_bounds = {'Tensor.d': 5e-9,
                    'Tensor.dperp0': 5e-9,
                    'Tensor.dperp1': 3e-9,
                    'CHARMEDRestricted0.d': 3e-9}

    inits = {'Tensor.d': 1.2e-9,
             'Tensor.dperp0': 0.5e-9,
             'Tensor.dperp1': 0.5e-9,
             'CHARMEDRestricted0.d': 1e-9}

    post_optimization_modifiers = [
        ('FR', lambda results: 1 - results['w_hin0.w'])
    ]


class CHARMED_r2(DMRICompositeModelConfig):

    description = 'The CHARMED model with 2 restricted compartments'

    model_expression = '''
        S0 * ( (Weight(w_hin0) * Tensor) +
               (Weight(w_res0) * CHARMEDRestricted(CHARMEDRestricted0)) +
               (Weight(w_res1) * CHARMEDRestricted(CHARMEDRestricted1)) )
    '''

    lower_bounds = {'Tensor.d': 1e-9,
                    'Tensor.dperp0': 0.3e-9,
                    'Tensor.dperp1': 0.3e-9,
                    'CHARMEDRestricted0.d': 0.3e-9,
                    'CHARMEDRestricted1.d': 0.3e-9}

    upper_bounds = {'Tensor.d': 5e-9,
                    'Tensor.dperp0': 5e-9,
                    'Tensor.dperp1': 3e-9,
                    'CHARMEDRestricted0.d': 3e-9,
                    'CHARMEDRestricted1.d': 3e-9}

    inits = {'Tensor.d': 1.2e-9,
             'Tensor.dperp0': 0.5e-9,
             'Tensor.dperp1': 0.5e-9,
             'CHARMEDRestricted0.d': 1e-9,
             'CHARMEDRestricted1.d': 1e-9,
             'w_res0.w': 0.1,
             'w_res1.w': 0.1}

    post_optimization_modifiers = [
        ('FR', lambda results: 1 - results['w_hin0.w'])
    ]

    prior = 'return w_res1.w < w_res0.w;'


class CHARMED_r3(DMRICompositeModelConfig):

    description = 'The standard CHARMED model with 3 restricted compartments'

    model_expression = '''
        S0 * ( (Weight(w_hin0) * Tensor) +
               (Weight(w_res0) * CHARMEDRestricted(CHARMEDRestricted0)) +
               (Weight(w_res1) * CHARMEDRestricted(CHARMEDRestricted1)) +
               (Weight(w_res2) * CHARMEDRestricted(CHARMEDRestricted2)) )
    '''

    lower_bounds = {'Tensor.d': 1e-9,
                    'Tensor.dperp0': 0.3e-9,
                    'Tensor.dperp1': 0.3e-9,
                    'CHARMEDRestricted0.d': 0.3e-9,
                    'CHARMEDRestricted1.d': 0.3e-9,
                    'CHARMEDRestricted2.d': 0.3e-9}

    upper_bounds = {'Tensor.d': 5e-9,
                    'Tensor.dperp0': 5e-9,
                    'Tensor.dperp1': 3e-9,
                    'CHARMEDRestricted0.d': 3e-9,
                    'CHARMEDRestricted1.d': 3e-9,
                    'CHARMEDRestricted2.d': 3e-9}

    inits = {'Tensor.d': 1.2e-9,
             'Tensor.dperp0': 0.5e-9,
             'Tensor.dperp1': 0.5e-9,
             'CHARMEDRestricted0.d': 1e-9,
             'CHARMEDRestricted1.d': 1e-9,
             'CHARMEDRestricted2.d': 1e-9,
             'w_res0.w': 0.1,
             'w_res1.w': 0.1,
             'w_res2.w': 0.1}

    post_optimization_modifiers = [
        ('FR', lambda results: 1 - results['w_hin0.w'])
    ]

    prior = 'return w_res2.w < w_res1.w && w_res1.w < w_res0.w;'
