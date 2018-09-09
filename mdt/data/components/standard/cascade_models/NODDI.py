from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'NODDI')
    inits = {'NODDI': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_csf.w': 'w_ball.w',
                       'NODDI_IC.theta': 'Stick0.theta',
                       'NODDI_IC.phi': 'Stick0.phi'}}


class NODDIDA(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'NODDIDA')
    inits = {'NODDIDA': {'NODDI_IC.theta': 'Stick0.theta',
                         'NODDI_IC.phi': 'Stick0.phi'}}


class NODDI_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'NODDI')
    inits = {'NODDI': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_csf.w': 'w_ball.w'}}
    fixes = {'NODDI': {'NODDI_IC.theta': 'Stick0.theta',
                       'NODDI_IC.phi': 'Stick0.phi'}}



class BinghamNODDI(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'BinghamNODDI')
    inits = {'BinghamNODDI': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                              'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                              'w_csf.w': 'w_ball.w',
                              'BinghamNODDI_IN.theta': 'Stick0.theta',
                              'BinghamNODDI_IN.phi': 'Stick0.phi'}}


class BinghamNODDI_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'BinghamNODDI')
    inits = {'BinghamNODDI': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                              'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                              'w_csf.w': 'w_ball.w'}}
    fixes = {'BinghamNODDI': {'BinghamNODDI_IN.theta': 'Stick0.theta',
                              'BinghamNODDI_IN.phi': 'Stick0.phi'}}


