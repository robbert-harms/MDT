from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallStick_r1(CascadeTemplate):

    description = 'Cascade for Ballstick'
    models = ('S0',
              'BallStick_r1')


class BallStick_r2(CascadeTemplate):

    description = 'Cascade for BallStick_r2.'
    models = ('BallStick_r1 (Cascade)',
              'BallStick_r2')
    inits = {'BallStick_r2': {'w_stick1.w': 0.0}}


class BallStick_r3(CascadeTemplate):

    description = 'Cascade for BallStick_r3.'
    models = ('BallStick_r2 (Cascade)',
              'BallStick_r3')
    inits = {'BallStick_r3': {'w_stick2.w': 0.0}}
