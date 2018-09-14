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


class NODDI_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'NODDI')
    inits = {'NODDI': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                       'w_csf.w': 'w_ball.w'}}
    fixes = {'NODDI': {'NODDI_IC.theta': 'Stick0.theta',
                       'NODDI_IC.phi': 'Stick0.phi'}}


class NODDI_ExVivo(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'NODDI_ExVivo')
    inits = {'NODDI_ExVivo': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                              'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                              'w_csf.w': 'w_ball.w',
                              'NODDI_IC.theta': 'Stick0.theta',
                              'NODDI_IC.phi': 'Stick0.phi'}}


class NODDI_ExVivo_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'NODDI_ExVivo')
    inits = {'NODDI_ExVivo': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                              'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                              'w_csf.w': 'w_ball.w'}}
    fixes = {'NODDI_ExVivo': {'NODDI_IC.theta': 'Stick0.theta',
                              'NODDI_IC.phi': 'Stick0.phi'}}



class NODDIDA(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'NODDIDA')
    inits = {'NODDIDA': {'NODDI_IC.theta': 'Stick0.theta',
                         'NODDI_IC.phi': 'Stick0.phi'}}


class BinghamNODDI_r1(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'BinghamNODDI_r1')
    inits = {'BinghamNODDI_r1': {'w_in0.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                                 'w_en0.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                                 'w_csf.w': 'w_ball.w',
                                 'BinghamNODDI_IN0.theta': 'Stick0.theta',
                                 'BinghamNODDI_IN0.phi': 'Stick0.phi'}}


class BinghamNODDI_r1_fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'BinghamNODDI_r1')
    inits = {'BinghamNODDI_r1': {'w_in0.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                                 'w_en0.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                                 'w_csf.w': 'w_ball.w'}}
    fixes = {'BinghamNODDI': {'BinghamNODDI_IN0.theta': 'Stick0.theta',
                              'BinghamNODDI_IN0.phi': 'Stick0.phi'}}


class BinghamNODDI_r2(CascadeTemplate):

    models = ('BallStick_r2 (Cascade)',
              'BinghamNODDI_r1 (Cascade)',
              'BinghamNODDI_r2')
    inits = {'BinghamNODDI_r2': {'BinghamNODDI_IN1.theta': lambda _, out_all: out_all[0]['Stick1.theta'],
                                 'BinghamNODDI_IN1.phi': lambda _, out_all: out_all[0]['Stick1.phi']}}
