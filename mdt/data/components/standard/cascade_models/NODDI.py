from mdt.models.cascade import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI(CascadeConfig):

    name = 'NODDI (Cascade)'
    description = 'Cascade for NODDI initialized from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'NODDI')
    inits = {'NODDI': [('w_ic.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'] / 2.0),
                       ('w_ec.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w'),
                       ('NODDI_IC.theta', 'Stick.theta'),
                       ('NODDI_IC.phi', 'Stick.phi')]}


class NODDI_S0(CascadeConfig):

    name = 'NODDI (Cascade|S0)'
    description = 'Cascade for NODDI initialized with only an S0 fit.'
    models = ('S0',
              'NODDI')


class NODDI_Fixed(CascadeConfig):

    name = 'NODDI (Cascade|fixed)'
    description = 'Cascade for NODDI with fixed directions from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'NODDI')
    inits = {'NODDI': [('w_ic.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'] / 2.0),
                       ('w_ec.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w')]}
    fixes = {'NODDI': [('NODDI_IC.theta', 'Stick.theta'),
                       ('NODDI_IC.phi', 'Stick.phi')]}


class NODDI_2(CascadeConfig):

    name = 'NODDI2 (Cascade)'
    description = 'Cascade for NODDI2 initialized from Ball & 2x Stick.'
    models = ('BallStickStick (Cascade)',
              'NODDI2')
    inits = {'NODDI': [('w_ic0.w', lambda output_previous, output_all_previous: output_previous['w_stick0.w'] / 2.0),
                       ('w_ec0.w', lambda output_previous, output_all_previous: output_previous['w_stick0.w'] / 2.0),
                       ('w_ic1.w', lambda output_previous, output_all_previous: output_previous['w_stick1.w'] / 2.0),
                       ('w_ec1.w', lambda output_previous, output_all_previous: output_previous['w_stick1.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w'),
                       ('NODDI_IC0.theta', 'Stick0.theta'),
                       ('NODDI_IC0.phi', 'Stick0.phi'),
                       ('NODDI_IC1.theta', 'Stick1.theta'),
                       ('NODDI_IC1.phi', 'Stick1.phi')]}


class NODDI2_Fixed(CascadeConfig):

    name = 'NODDI2 (Cascade|fixed)'
    description = 'Cascade for NODDI2 with fixed directions from Ball & 2x Stick.'
    models = ('BallStickStick (Cascade)',
              'NODDI2')
    inits = {'NODDI': [('w_ic0.w', lambda output_previous, output_all_previous: output_previous['w_stick0.w'] / 2.0),
                       ('w_ec0.w', lambda output_previous, output_all_previous: output_previous['w_stick0.w'] / 2.0),
                       ('w_ic1.w', lambda output_previous, output_all_previous: output_previous['w_stick1.w'] / 2.0),
                       ('w_ec1.w', lambda output_previous, output_all_previous: output_previous['w_stick1.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w')]}
    fixes = {'NODDI': [('NODDI_IC0.theta', 'Stick0.theta'),
                       ('NODDI_IC0.phi', 'Stick0.phi'),
                       ('NODDI_IC1.theta', 'Stick1.theta'),
                       ('NODDI_IC1.phi', 'Stick1.phi')]}
