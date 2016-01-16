from mdt.models.cascade import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Noddi(CascadeConfig):

    name = 'Noddi (Cascade)'
    description = 'Cascade for Noddi initialized from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'Noddi')
    inits = {'Noddi': [('w_ic.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'] / 2.0),
                       ('w_ec.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w'),
                       ('Noddi_IC.theta', 'Stick.theta'),
                       ('Noddi_IC.phi', 'Stick.phi')]}


class NoddiS0(CascadeConfig):

    name = 'Noddi (Cascade|S0)'
    description = 'Cascade for Noddi initialized with only an S0 fit.'
    models = ('S0',
              'Noddi')


class NoddiFixed(CascadeConfig):

    name = 'Noddi (Cascade|fixed)'
    description = 'Cascade for Noddi with fixed directions from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'Noddi')
    inits = {'Noddi': [('w_ic.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'] / 2.0),
                       ('w_ec.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w')]}
    fixes = {'Noddi': [('Noddi_IC.theta', 'Stick.theta'),
                       ('Noddi_IC.phi', 'Stick.phi')]}


class Noddi2(CascadeConfig):

    name = 'Noddi2 (Cascade)'
    description = 'Cascade for Noddi2 initialized from Ball & 2x Stick.'
    models = ('BallStickStick (Cascade)',
              'Noddi2')
    inits = {'Noddi': [('w_ic0.w', lambda output_previous, output_all_previous: output_previous['w_stick0.w'] / 2.0),
                       ('w_ec0.w', lambda output_previous, output_all_previous: output_previous['w_stick0.w'] / 2.0),
                       ('w_ic1.w', lambda output_previous, output_all_previous: output_previous['w_stick1.w'] / 2.0),
                       ('w_ec1.w', lambda output_previous, output_all_previous: output_previous['w_stick1.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w'),
                       ('Noddi_IC0.theta', 'Stick0.theta'),
                       ('Noddi_IC0.phi', 'Stick0.phi'),
                       ('Noddi_IC1.theta', 'Stick1.theta'),
                       ('Noddi_IC1.phi', 'Stick1.phi')]}


class Noddi2Fixed(CascadeConfig):

    name = 'Noddi2 (Cascade|fixed)'
    description = 'Cascade for Noddi2 with fixed directions from Ball & 2x Stick.'
    models = ('BallStickStick (Cascade)',
              'Noddi2')
    inits = {'Noddi': [('w_ic0.w', lambda output_previous, output_all_previous: output_previous['w_stick0.w'] / 2.0),
                       ('w_ec0.w', lambda output_previous, output_all_previous: output_previous['w_stick0.w'] / 2.0),
                       ('w_ic1.w', lambda output_previous, output_all_previous: output_previous['w_stick1.w'] / 2.0),
                       ('w_ec1.w', lambda output_previous, output_all_previous: output_previous['w_stick1.w'] / 2.0),
                       ('w_csf.w', 'w_ball.w')]}
    fixes = {'Noddi': [('Noddi_IC0.theta', 'Stick0.theta'),
                       ('Noddi_IC0.phi', 'Stick0.phi'),
                       ('Noddi_IC1.theta', 'Stick1.theta'),
                       ('Noddi_IC1.phi', 'Stick1.phi')]}
