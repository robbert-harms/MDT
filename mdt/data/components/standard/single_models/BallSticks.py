from mdt.models.single import DMRISingleModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallStick(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The default Ball & Stick model'
    model_expression = '''
        S0 * ( (Weight(w_ball) * Ball) +
               (Weight(w_stick) * Stick) )
    '''
    fixes = {'Ball.d': 3.0e-9,
             'Stick.d': 1.7e-9}
    post_optimization_modifiers = [('FS', lambda results: 1 - results['w_ball.w'])]


class BallStickT2(BallStick):

    name = 'BallStick-T2'
    description = 'The Ball & Stick model with extra T2 weighting'
    model_expression = '''
        S0 * ExpT2Dec * ( (Weight(w_ball) * Ball) +
                          (Weight(w_stick) * Stick) )
    '''


class BallStickT2T2(BallStick):

    name = 'BallStick-T2T2'
    description = 'The Ball & Stick model with two extra T2 models'
    model_expression = '''
            S0 * ( (ExpT2Dec(T2_long) * Weight(w_ball) * Ball) +
                   (ExpT2Dec(T2_short) * Weight(w_stick) * Stick) )
        '''


class BallStickExVivo(BallStick):

    name = 'BallStick-ExVivo'
    in_vivo_suitable = False
    ex_vivo_suitable = True
    description = 'The Ball & Stick model with ex vivo defaults',
    fixes = {'Ball.d': 2.0e-9,
             'Stick.d': 0.6e-9}


class StickExVivo(DMRISingleModelConfig):

    name = 'Stick-ExVivo'
    in_vivo_suitable = False
    ex_vivo_suitable = True
    description = 'The Stick model with ex vivo defaults',
    fixes = {'Stick.d': 0.6e-9}
    model_expression = '''
        S0 * Stick
    '''


class BallStickT2ExVivo(BallStickExVivo):

    name = 'BallStick-T2-ExVivo'
    description = 'The Ball & Stick model with extra T2 weighting and exvivo defaults'
    model_expression = '''
            S0 * ExpT2Dec * ( (Weight(w_ball) * Ball) +
                              (Weight(w_stick) * Stick) )
        '''


class StickStickExVivo(DMRISingleModelConfig):

    name = 'StickStick-ExVivo'
    in_vivo_suitable = False
    ex_vivo_suitable = True
    description = 'The 2x Stick model ex vivo defaults'
    model_expression = '''
            S0 * ( (Weight(w_stick0) * Stick(Stick0)) +
                   (Weight(w_stick1) * Stick(Stick1)) )
        '''
    fixes = {'Stick0.d': 1.7e-9,
             'Stick1.d': 1.7e-9}


class BallStickStick(DMRISingleModelConfig):

    ex_vivo_suitable = False
    description = 'The Ball & 2x Stick model'
    model_expression = '''
            S0 * ( (Weight(w_ball) * Ball) +
                   (Weight(w_stick0) * Stick(Stick0)) +
                   (Weight(w_stick1) * Stick(Stick1)) )
        '''
    fixes = {'Ball.d': 3.0e-9,
             'Stick0.d': 1.7e-9,
             'Stick1.d': 1.7e-9}
    post_optimization_modifiers = [('FS', lambda results: 1 - results['w_ball.w'])]


class BallStickStickExVivo(BallStickStick):

    name = 'BallStickStick-ExVivo'
    in_vivo_suitable = False
    ex_vivo_suitable = True
    description = 'The Ball & 2x Stick model with ex vivo defaults'
    fixes = {'Ball.d': 2.0e-9,
             'Stick0.d': 0.6e-9,
             'Stick1.d': 0.6e-9}


class BallStickStickStick(DMRISingleModelConfig):

    ex_vivo_suitable = False
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

    post_optimization_modifiers = [('FS', lambda results: 1 - results['w_ball.w'])]


class BallStickStickStickExVivo(BallStickStickStick):

    name = 'BallStickStickStick-ExVivo'
    ex_vivo_suitable = True
    in_vivo_suitable = False
    description = 'The Ball & 3x Stick model with ex vivo defaults'
    fixes = {'Ball.d': 2.0e-9,
             'Stick0.d': 0.6e-9,
             'Stick1.d': 0.6e-9,
             'Stick2.d': 0.6e-9}
