from mdt.models.cascade import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallStick(CascadeConfig):

    name = 'BallStick (Cascade)'
    description = 'Cascade for Ballstick'
    models = ('S0',
              'BallStick')


class BallStickS0(BallStick):

    name = 'BallStick (Cascade|S0)'


class BallStickExVivo(BallStick):

    name = 'BallStick-ExVivo (Cascade)'
    description = 'Cascade for Ballstick with ex vivo defaults.'
    models = ('S0',
              'BallStick-ExVivo')


class BallStickStick(CascadeConfig):

    name = 'BallStickStick (Cascade)'
    description = 'Cascade for BallStickStick.'
    models = ('BallStick (Cascade)',
              'BallStickStick')
    inits = {'BallStickStick': [('Stick0.theta', 'Stick.theta'),
                                ('Stick0.phi', 'Stick.phi'),
                                ('w_stick0.w', 'w_stick.w'),
                                ('w_stick1.w', 0.0)]}


class BallStickStickExVivo(BallStickStick):

    name = 'BallStickStick-ExVivo (Cascade)'
    description = 'Cascade for BallStickStick with ex vivo defaults.'
    models = ('BallStick-ExVivo (Cascade)',
              'BallStickStick-ExVivo')


class BallStickStickStick(CascadeConfig):

    name = 'BallStickStickStick (Cascade)'
    description = 'Cascade for BallStickStickStick.'
    models = ('BallStickStick (Cascade)',
              'BallStickStickStick')
    inits = {'BallStickStickStick': [('w_stick2.w', 0.0)]}


class BallStickStickStickExVivo(BallStickStickStick):

    name = 'BallStickStickStick-ExVivo (Cascade)'
    description = 'Cascade for BallStickStickStick with ex vivo defaults.'
    models = ('BallStickStick-ExVivo (Cascade)',
              'BallStickStickStick-ExVivo')
