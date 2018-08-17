from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI(CascadeTemplate):

    description = 'Cascade for NODDI initialized from Ball&Stick.'
    models = ('BallStick_r1 (Cascade)',
              'NODDI')
    inits = {'NODDI': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_csf.w': 'w_ball.w',
                       'NODDI_IC.theta': 'Stick0.theta',
                       'NODDI_IC.phi': 'Stick0.phi'}}


class NODDIDA(CascadeTemplate):

    description = 'Cascade for NODDIDA initialized from Ball&Stick.'
    models = ('BallStick_r1 (Cascade)',
              'NODDIDA')
    inits = {'NODDIDA': {'NODDI_IC.theta': 'Stick0.theta',
                         'NODDI_IC.phi': 'Stick0.phi'}}


class NODDI_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    description = 'Cascade for NODDI with fixed directions from Ball&Stick.'
    models = ('BallStick_r1 (Cascade)',
              'NODDI')
    inits = {'NODDI': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_csf.w': 'w_ball.w'}}
    fixes = {'NODDI': {'NODDI_IC.theta': 'Stick0.theta',
                       'NODDI_IC.phi': 'Stick0.phi'}}
