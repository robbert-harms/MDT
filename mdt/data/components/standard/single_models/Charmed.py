from mdt.models.single import DMRISingleModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Charmed_r1(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The Charmed (CHARMED) model with 1 restricted compartments'

    model_expression = '''
        S0 * ( (Weight(w_hin0) * Tensor) +
               (Weight(w_res0) * CharmedRestricted(CharmedRestricted0))
               )
    '''

    lower_bounds = {'Tensor.d': 1e-9,
                    'Tensor.dperp0': 0.3e-9,
                    'Tensor.dperp1': 0.3e-9,
                    'CharmedRestricted0.d': 0.3e-9}

    upper_bounds = {'Tensor.d': 5e-9,
                    'Tensor.dperp0': 5e-9,
                    'Tensor.dperp1': 3e-9,
                    'CharmedRestricted0.d': 3e-9}

    inits = {'Tensor.d': 1.2e-9,
             'Tensor.dperp0': 0.5e-9,
             'Tensor.dperp1': 0.5e-9,
             'CharmedRestricted0.d': 1e-9}

    post_optimization_modifiers = [
        ('FR', lambda results: 1 - results['w_hin0.w'])
    ]


class Charmed_r2(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The Charmed (CHARMED) model with 2 restricted compartments'

    model_expression = '''
        S0 * ( (Weight(w_hin0) * Tensor) +
               (Weight(w_res0) * CharmedRestricted(CharmedRestricted0)) +
               (Weight(w_res1) * CharmedRestricted(CharmedRestricted1)) )
    '''

    lower_bounds = {'Tensor.d': 1e-9,
                    'Tensor.dperp0': 0.3e-9,
                    'Tensor.dperp1': 0.3e-9,
                    'CharmedRestricted0.d': 0.3e-9,
                    'CharmedRestricted1.d': 0.3e-9}

    upper_bounds = {'Tensor.d': 5e-9,
                    'Tensor.dperp0': 5e-9,
                    'Tensor.dperp1': 3e-9,
                    'CharmedRestricted0.d': 3e-9,
                    'CharmedRestricted1.d': 3e-9}

    inits = {'Tensor.d': 1.2e-9,
             'Tensor.dperp0': 0.5e-9,
             'Tensor.dperp1': 0.5e-9,
             'CharmedRestricted0.d': 1e-9,
             'CharmedRestricted1.d': 1e-9}

    post_optimization_modifiers = [
        ('FR', lambda results: 1 - results['w_hin0.w'])
    ]


class Charmed(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The standard Charmed (CHARMED) model with 3 restricted compartments'

    model_expression = '''
        S0 * ( (Weight(w_hin0) * Tensor) +
               (Weight(w_res0) * CharmedRestricted(CharmedRestricted0)) +
               (Weight(w_res1) * CharmedRestricted(CharmedRestricted1)) +
               (Weight(w_res2) * CharmedRestricted(CharmedRestricted2)) )
    '''

    lower_bounds = {'Tensor.d': 1e-9,
                    'Tensor.dperp0': 0.3e-9,
                    'Tensor.dperp1': 0.3e-9,
                    'CharmedRestricted0.d': 0.3e-9,
                    'CharmedRestricted1.d': 0.3e-9,
                    'CharmedRestricted2.d': 0.3e-9}

    upper_bounds = {'Tensor.d': 5e-9,
                    'Tensor.dperp0': 5e-9,
                    'Tensor.dperp1': 3e-9,
                    'CharmedRestricted0.d': 3e-9,
                    'CharmedRestricted1.d': 3e-9,
                    'CharmedRestricted2.d': 3e-9}

    inits = {'Tensor.d': 1.2e-9,
             'Tensor.dperp0': 0.5e-9,
             'Tensor.dperp1': 0.5e-9,
             'CharmedRestricted0.d': 1e-9,
             'CharmedRestricted1.d': 1e-9,
             'CharmedRestricted2.d': 1e-9,
             'w_res2.w': 0}

    post_optimization_modifiers = [
        ('FR', lambda results: 1 - results['w_hin0.w'])
    ]


class Charmed_r3(Charmed):
    pass
