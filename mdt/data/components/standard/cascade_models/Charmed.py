from mdt.models.cascade import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CharmedR1(CascadeConfig):

    name = 'Charmed_r1 (Cascade)'
    description = 'Initializes the directions to Ball & Stick.'
    models = ('BallStick (Cascade)',
              'Charmed_r1')
    inits = {'Charmed_r1': [('CharmedRestricted0.theta', 'Stick.theta'),
                            ('CharmedRestricted0.phi', 'Stick.phi'),
                            ('Tensor.theta', 'Stick.theta'),
                            ('Tensor.phi', 'Stick.phi'),
                            ('w_res0.w', 'w_stick.w')]}


class CharmedR1S0(CascadeConfig):

    name = 'Charmed_r1 (Cascade|S0)'
    description = 'Cascade for Charmed r1 initialized with only an S0 fit.'
    models = ('S0',
              'Charmed_r1')


class CharmedR1Fixed(CascadeConfig):

    name = 'Charmed_r1 (Cascade|fixed)'
    description = 'Fixes the directions to Ball & Stick.'
    models = ('BallStick (Cascade)',
              'Charmed_r1')
    inits = {'Charmed_r1': [('Tensor.theta', 'Stick.theta'),
                            ('Tensor.phi', 'Stick.phi'),
                            ('w_res0.w', 'w_stick.w')]}
    fixes = {'Charmed_r1': [('CharmedRestricted0.theta', 'Stick.theta'),
                            ('CharmedRestricted0.phi', 'Stick.phi')]}


class CharmedR2(CascadeConfig):

    name = 'Charmed_r2 (Cascade)'
    description = 'Initializes the directions to 2x Ball & Stick.'
    models = ('BallStickStick (Cascade)',
              'Charmed_r2')
    inits = {'Charmed_r2': [('Tensor.theta', 'Stick0.theta'),
                            ('Tensor.phi', 'Stick0.phi'),
                            ('CharmedRestricted0.theta', 'Stick0.theta'),
                            ('CharmedRestricted0.phi', 'Stick0.phi'),
                            ('CharmedRestricted1.theta', 'Stick1.theta'),
                            ('CharmedRestricted1.phi', 'Stick1.phi'),
                            ('w_res0.w', 'w_stick0.w'),
                            ('w_res1.w', 'w_stick1.w')]}


class CharmedR2S0(CascadeConfig):

    name = 'Charmed_r2 (Cascade|S0)'
    description = 'Initializes with only an S0 fit.'
    models = ('S0',
              'Charmed_r2')


class CharmedR2Fixed(CascadeConfig):

    name = 'Charmed_r2 (Cascade|fixed)'
    description = 'Fixes the directions to 2x Ball & Stick.'
    models = ('BallStickStick (Cascade)',
              'Charmed_r2')
    inits = {'Charmed_r2': [('Tensor.theta', 'Stick0.theta'),
                            ('Tensor.phi', 'Stick0.phi'),
                            ('w_res0.w', 'w_stick0.w'),
                            ('w_res1.w', 'w_stick1.w')]}
    fixes = {'Charmed_r2': [('CharmedRestricted0.theta', 'Stick0.theta'),
                            ('CharmedRestricted0.phi', 'Stick0.phi'),
                            ('CharmedRestricted1.theta', 'Stick1.theta'),
                            ('CharmedRestricted1.phi', 'Stick1.phi'),
                            ]}


class Charmed(CascadeConfig):

    name = 'Charmed (Cascade)'
    description = 'Initializes the directions to 3x Ball & Stick.'
    models = ('BallStickStickStick (Cascade)',
              'Charmed')
    inits = {'Charmed': [('Tensor.theta', 'Stick0.theta'),
                         ('Tensor.phi', 'Stick0.phi'),
                         ('w_res0.w', 'w_stick0.w'),
                         ('w_res1.w', 'w_stick1.w'),
                         ('w_res2.w', 'w_stick2.w'),
                         ('CharmedRestricted0.theta', 'Stick0.theta'),
                         ('CharmedRestricted0.phi', 'Stick0.phi'),
                         ('CharmedRestricted1.theta', 'Stick1.theta'),
                         ('CharmedRestricted1.phi', 'Stick1.phi'),
                         ('CharmedRestricted2.theta', 'Stick2.theta'),
                         ('CharmedRestricted2.phi', 'Stick2.phi'),
                         ]}


class CharmedS0(CascadeConfig):

    name = 'Charmed (Cascade|S0)'
    description = 'Initializes with only an S0 fit.'
    models = ('S0',
              'Charmed')


class CharmedFixed(CascadeConfig):

    name = 'Charmed (Cascade|fixed)'
    description = 'Fixes the directions to 3x Ball & Stick.'
    models = ('BallStickStickStick (Cascade)',
              'Charmed')
    inits = {'Charmed': [('Tensor.theta', 'Stick0.theta'),
                         ('Tensor.phi', 'Stick0.phi'),
                         ('w_res0.w', 'w_stick0.w'),
                         ('w_res1.w', 'w_stick1.w'),
                         ('w_res2.w', 'w_stick2.w')]}
    fixes = {'Charmed': [('CharmedRestricted0.theta', 'Stick0.theta'),
                         ('CharmedRestricted0.phi', 'Stick0.phi'),
                         ('CharmedRestricted1.theta', 'Stick1.theta'),
                         ('CharmedRestricted1.phi', 'Stick1.phi'),
                         ('CharmedRestricted2.theta', 'Stick2.theta'),
                         ('CharmedRestricted2.phi', 'Stick2.phi')]}
