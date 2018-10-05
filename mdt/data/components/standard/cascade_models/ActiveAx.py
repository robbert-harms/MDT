from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-07-19'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ActiveAx(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'ActiveAx')
    inits = {'ActiveAx': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                          'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                          'w_csf.w': 'w_ball.w',
                          'CylinderGPD.theta': 'Stick0.theta',
                          'CylinderGPD.phi': 'Stick0.phi'}}


class ActiveAx_ExVivo(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'ActiveAx_ExVivo')
    inits = {'ActiveAx_ExVivo': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                                 'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                                 'w_csf.w': 'w_ball.w',
                                 'CylinderGPD.theta': 'Stick0.theta',
                                 'CylinderGPD.phi': 'Stick0.phi'}}


class ActiveAx_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'ActiveAx')
    fixes = {'ActiveAx': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                          'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                          'w_csf.w': 'w_ball.w',
                          'CylinderGPD.theta': 'Stick0.theta',
                          'CylinderGPD.phi': 'Stick0.phi'}}


class ActiveAx_ExVivo_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'ActiveAx_ExVivo')
    fixes = {'ActiveAx_ExVivo': {'w_ic.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                                 'w_ec.w': lambda output_previous, _: output_previous['w_stick0.w'] / 2.0,
                                 'w_csf.w': 'w_ball.w',
                                 'CylinderGPD.theta': 'Stick0.theta',
                                 'CylinderGPD.phi': 'Stick0.phi'}}
