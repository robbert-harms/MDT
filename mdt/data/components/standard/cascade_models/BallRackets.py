from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallRacket_r1_c(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'BallRacket_r1')
    inits = {'BallRacket_r1': {'Racket0.theta': 'Stick0.theta',
                               'Racket0.phi': 'Stick0.phi',
                               'w_res0.w': 'w_stick0.w'}}


class BallRacket_r2(CascadeTemplate):

    models = ('BallStick_r2 (Cascade)',
              'BallRacket_r1 (Cascade)',
              'BallRacket_r2')
    inits = {'BallRacket_r2': {
        'w_res1.w': 0.0,
        'Racket1.theta': lambda _, out_all: out_all[0]['Stick1.theta'],
        'Racket1.phi': lambda _, out_all: out_all[0]['Stick1.phi']}}


class BallRacket_r2_fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r2 (Cascade)',
              'BallRacket_r1 (Cascade)',
              'BallRacket_r2')
    inits = {'BallRacket_r2': {'w_res1.w': 0.0}}
    fixes = {'BallRacket_r2': {
        'Racket0.theta': lambda _, out_all: out_all[0]['Stick0.theta'],
        'Racket0.phi': lambda _, out_all: out_all[0]['Stick0.phi'],
        'Racket1.theta': lambda _, out_all: out_all[0]['Stick1.theta'],
        'Racket1.phi': lambda _, out_all: out_all[0]['Stick1.phi']
    }}
