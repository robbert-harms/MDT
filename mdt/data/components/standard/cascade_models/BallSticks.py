from mdt.components_config.cascade_models import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallStick_r1(CascadeConfig):

    name = 'BallStick_r1 (Cascade)'
    description = 'Cascade for Ballstick'
    models = ('S0',
              'BallStick_r1')


class BallStick_r1_ExVivo(BallStick_r1):

    name = 'BallStick_r1-ExVivo (Cascade)'
    description = 'Cascade for Ballstick with ex vivo defaults.'
    models = ('S0',
              'BallStick_r1-ExVivo')


class BallStick_r2(CascadeConfig):

    name = 'BallStick_r2 (Cascade)'
    description = 'Cascade for BallStick_r2.'
    models = ('BallStick_r1 (Cascade)',
              'BallStick_r2')
    inits = {'BallStick_r2': [('Stick0.theta', 'Stick.theta'),
                              ('Stick0.phi', 'Stick.phi'),
                              ('w_stick0.w', 'w_stick.w'),
                              ('w_stick1.w', 0.0)]}


class BallStick_r2_ExVivo(BallStick_r2):

    name = 'BallStick_r2-ExVivo (Cascade)'
    description = 'Cascade for BallStick_r2 with ex vivo defaults.'
    models = ('BallStick_r1-ExVivo (Cascade)',
              'BallStick_r2-ExVivo')


class BallStick_r3(CascadeConfig):

    name = 'BallStick_r3 (Cascade)'
    description = 'Cascade for BallStick_r3.'
    models = ('BallStick_r2 (Cascade)',
              'BallStick_r3')
    inits = {'BallStick_r3': [('w_stick2.w', 0.0)]}


class BallStick_r3_ExVivo(BallStick_r3):

    name = 'BallStick_r3-ExVivo (Cascade)'
    description = 'Cascade for BallStick_r3 with ex vivo defaults.'
    models = ('BallStick_r2-ExVivo (Cascade)',
              'BallStick_r3-ExVivo')
