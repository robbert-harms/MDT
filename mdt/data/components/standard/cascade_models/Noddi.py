from mdt.models.cascade import CascadeModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Noddi(CascadeModelBuilder):

    name = 'Noddi (Cascade)'
    description = 'Cascade for Noddi initialized from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'Noddi')
    inits = {'Noddi': [('Wic.w', lambda output_previous: output_previous['Wstick.w']/2.0),
                       ('Wec.w', lambda output_previous: output_previous['Wstick.w']/2.0),
                       ('Wcsf.w', 'Wball.w'),
                       ('Noddi_IC.theta', 'Stick.theta'),
                       ('Noddi_IC.phi', 'Stick.phi')]}


class NoddiS0(CascadeModelBuilder):

    name = 'Noddi (Cascade|s0)'
    description = 'Cascade for Noddi initialized with only an S0 fit.'
    models = ('s0',
              'Noddi')


class NoddiFixed(CascadeModelBuilder):

    name = 'Noddi (Cascade|fixed)'
    description = 'Cascade for Noddi with fixed directions from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'Noddi')
    inits = {'Noddi': [('Wic.w', lambda output_previous: output_previous['Wstick.w']/2.0),
                       ('Wec.w', lambda output_previous: output_previous['Wstick.w']/2.0),
                       ('Wcsf.w', 'Wball.w')]}
    fixes = {'Noddi': [('Noddi_IC.theta', 'Stick.theta'),
                       ('Noddi_IC.phi', 'Stick.phi')]}


class Noddi2(CascadeModelBuilder):

    name = 'Noddi2 (Cascade)'
    description = 'Cascade for Noddi2 initialized from Ball & 2x Stick.'
    models = ('BallStickStick (Cascade)',
              'Noddi2')

    inits = {'Noddi': [('Wic0.w', lambda output_previous: output_previous['Wstick0.w']/2.0),
                       ('Wec0.w', lambda output_previous: output_previous['Wstick0.w']/2.0),
                       ('Wic1.w', lambda output_previous: output_previous['Wstick1.w']/2.0),
                       ('Wec1.w', lambda output_previous: output_previous['Wstick1.w']/2.0),
                       ('Wcsf.w', 'Wball.w'),
                       ('Noddi_IC0.theta', 'Stick0.theta'),
                       ('Noddi_IC0.phi', 'Stick0.phi'),
                       ('Noddi_IC1.theta', 'Stick1.theta'),
                       ('Noddi_IC1.phi', 'Stick1.phi')]}


class Noddi2Fixed(CascadeModelBuilder):

    name = 'Noddi2 (Cascade|fixed)'
    description = 'Cascade for Noddi2 with fixed directions from Ball & 2x Stick.'
    models = ('BallStickStick (Cascade)',
              'Noddi2')

    inits = {'Noddi': [('Wic0.w', lambda output_previous: output_previous['Wstick0.w']/2.0),
                       ('Wec0.w', lambda output_previous: output_previous['Wstick0.w']/2.0),
                       ('Wic1.w', lambda output_previous: output_previous['Wstick1.w']/2.0),
                       ('Wec1.w', lambda output_previous: output_previous['Wstick1.w']/2.0),
                       ('Wcsf.w', 'Wball.w')]}
    fixes = {'Noddi': [('Noddi_IC0.theta', 'Stick0.theta'),
                       ('Noddi_IC0.phi', 'Stick0.phi'),
                       ('Noddi_IC1.theta', 'Stick1.theta'),
                       ('Noddi_IC1.phi', 'Stick1.phi')]}
