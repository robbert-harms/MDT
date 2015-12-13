from mdt.models.single import DMRISingleModelBuilder


__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallStick(DMRISingleModelBuilder):

    config = dict(
        name='BallStick',
        ex_vivo_suitable=False,
        description='The default Ball & Stick model',
        model_expression='''
            S0 * ( (Weight(Wball) * Ball) +
                   (Weight(Wstick) * Stick) )
        ''',
        fixes={'Ball.d': 3.0e-9,
               'Stick.d': 1.7e-9},
        post_optimization_modifiers=[('SNIF', lambda results: 1 - results['Wball.w'])]
    )


class BallStickT2(BallStick):

    config = dict(
        name='BallStick-T2',
        description='The Ball & Stick model with extra T2 weighting',
        model_expression='''
            S0 * ExpT2Dec * ( (Weight(Wball) * Ball) +
                              (Weight(Wstick) * Stick) )
        '''
    )


class BallStickT2T2(BallStick):

    config = dict(
        name='BallStick-T2T2',
        description='The Ball & Stick model with two extra T2 models',
        model_expression='''
            S0 * ( (ExpT2Dec(T2long) * Weight(Wball) * Ball) +
                   (ExpT2Dec(T2short) * Weight(Wstick) * Stick) )
        '''
    )


class BallStickExVivo(BallStick):

    config = dict(
        name='BallStick-ExVivo',
        in_vivo_suitable=False,
        ex_vivo_suitable=True,
        description='The Ball & Stick model with ex vivo defaults',
        fixes={'Ball.d': 2.0e-9,
               'Stick.d': 0.6e-9}
    )


class BallStickT2ExVivo(BallStickExVivo):

    config = dict(
        name='BallStick-T2-ExVivo',
        description='The Ball & Stick model with extra T2 weighting and exvivo defaults',
        model_expression='''
            S0 * ExpT2Dec * ( (Weight(Wball) * Ball) +
                              (Weight(Wstick) * Stick) )
        '''
    )


class BallStickStick(DMRISingleModelBuilder):

    config = dict(
        name='BallStickStick',
        ex_vivo_suitable=False,
        description='The Ball & 2x Stick model',
        model_expression='''
            S0 * ( (Weight(Wball) * Ball) +
                   (Weight(Wstick0) * Stick(Stick0)) +
                   (Weight(Wstick1) * Stick(Stick1)) )
        ''',
        fixes={'Ball.d': 3.0e-9,
               'Stick0.d': 1.7e-9,
               'Stick1.d': 1.7e-9},
        post_optimization_modifiers=[('SNIF', lambda results: 1 - results['Wball.w'])]
    )


class BallStickStickExVivo(BallStickStick):

    config = dict(
        name='BallStickStick-ExVivo',
        in_vivo_suitable=False,
        ex_vivo_suitable=True,
        description='The Ball & 2x Stick model with ex vivo defaults',
        fixes={'Ball.d': 2.0e-9,
               'Stick0.d': 0.6e-9,
               'Stick1.d': 0.6e-9}
    )


class BallStickStickStick(DMRISingleModelBuilder):

    config = dict(
        name='BallStickStickStick',
        ex_vivo_suitable=False,
        description='The Ball & 3x Stick model',
        model_expression='''
            S0 * ( (Weight(Wball) * Ball) +
                   (Weight(Wstick0) * Stick(Stick0)) +
                   (Weight(Wstick1) * Stick(Stick1)) +
                   (Weight(Wstick2) * Stick(Stick2)) )
        ''',
        fixes={'Ball.d': 3.0e-9,
               'Stick0.d': 1.7e-9,
               'Stick1.d': 1.7e-9,
               'Stick2.d': 1.7e-9},
        post_optimization_modifiers=[('SNIF', lambda results: 1 - results['Wball.w'])]
    )


class BallStickStickStickExVivo(BallStickStickStick):

    config = dict(
        name='BallStickStickStick-ExVivo',
        ex_vivo_suitable=True,
        in_vivo_suitable=False,
        description='The Ball & 3x Stick model with ex vivo defaults',
        fixes={'Ball.d': 2.0e-9,
               'Stick0.d': 0.6e-9,
               'Stick1.d': 0.6e-9,
               'Stick2.d': 0.6e-9}
    )
