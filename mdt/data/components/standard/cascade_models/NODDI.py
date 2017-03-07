from mdt.components_config.cascade_models import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI(CascadeConfig):

    name = 'NODDI (Cascade)'
    description = 'Cascade for NODDI initialized from Ball&Stick.'
    models = ('BallStick_r1 (Cascade)',
              'NODDI')
    inits = {'NODDI': [('w_ic.w', lambda output_previous, _: output_previous['w_stick.w'] / 2.0),
                       ('w_ec.w', lambda output_previous, _: output_previous['w_stick.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w'),
                       ('NODDI_IC.theta', 'Stick.theta'),
                       ('NODDI_IC.phi', 'Stick.phi')]}


class NODDI_Fixed(CascadeConfig):

    name = 'NODDI (Cascade|fixed)'
    description = 'Cascade for NODDI with fixed directions from Ball&Stick.'
    models = ('BallStick_r1 (Cascade)',
              'NODDI')
    inits = {'NODDI': [('w_ic.w', lambda output_previous, _: output_previous['w_stick.w'] / 2.0),
                       ('w_ec.w', lambda output_previous, _: output_previous['w_stick.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w')]}
    fixes = {'NODDI': [('NODDI_IC.theta', 'Stick.theta'),
                       ('NODDI_IC.phi', 'Stick.phi')]}
