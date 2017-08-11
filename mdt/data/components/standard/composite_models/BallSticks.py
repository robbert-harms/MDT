from mdt.component_templates.composite_models import DMRICompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallStick_r1(DMRICompositeModelTemplate):

    description = 'The default Ball & Stick model'
    model_expression = '''
        S0 * ( (Weight(w_ball) * Ball) +
               (Weight(w_stick0) * Stick(Stick0)) )
    '''
    fixes = {'Ball.d': 3.0e-9,
             'Stick0.d': 1.7e-9}
    post_optimization_modifiers = [('FS', lambda results: 1 - results['w_ball.w'])]


class BallStick_r2(DMRICompositeModelTemplate):

    description = 'The Ball & 2x Stick model'
    model_expression = '''
        S0 * ( (Weight(w_ball) * Ball) +
               (Weight(w_stick0) * Stick(Stick0)) +
               (Weight(w_stick1) * Stick(Stick1)) )
    '''
    fixes = {'Ball.d': 3.0e-9,
             'Stick0.d': 1.7e-9,
             'Stick1.d': 1.7e-9}

    sort_maps = [('w_stick0', 'w_stick1'), ('Stick0', 'Stick1')]
    post_optimization_modifiers = [('FS', lambda results: 1 - results['w_ball.w'])]

    extra_prior = 'return w_stick1.w < w_stick0.w;'


class BallStick_r3(DMRICompositeModelTemplate):

    description = 'The Ball & 3x Stick model'
    model_expression = '''
            S0 * ( (Weight(w_ball) * Ball) +
                   (Weight(w_stick0) * Stick(Stick0)) +
                   (Weight(w_stick1) * Stick(Stick1)) +
                   (Weight(w_stick2) * Stick(Stick2)) )
        '''
    fixes = {'Ball.d': 3.0e-9,
             'Stick0.d': 1.7e-9,
             'Stick1.d': 1.7e-9,
             'Stick2.d': 1.7e-9}
    inits = {'w_stick2.w': 0}

    sort_maps = [('w_stick0', 'w_stick1', 'w_stick2'), ('Stick0', 'Stick1', 'Stick2')]
    post_optimization_modifiers = [('FS', lambda results: 1 - results['w_ball.w'])]

    extra_prior = 'return w_stick2.w < w_stick1.w && w_stick1.w < w_stick0.w;'
