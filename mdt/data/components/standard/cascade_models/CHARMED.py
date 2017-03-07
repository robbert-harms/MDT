from mdt.components_config.cascade_models import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CHARMED1(CascadeConfig):

    name = 'CHARMED_r1 (Cascade)'
    description = 'Initializes the directions to Ball & Stick.'
    models = ('BallStick_r1 (Cascade)',
              'CHARMED_r1')
    inits = {'CHARMED_r1': [('CHARMEDRestricted0.theta', 'Stick.theta'),
                            ('CHARMEDRestricted0.phi', 'Stick.phi'),
                            ('Tensor.theta', 'Stick.theta'),
                            ('Tensor.phi', 'Stick.phi'),
                            ('w_res0.w', 'w_stick.w')]}


class CHARMEDR1Fixed(CascadeConfig):

    name = 'CHARMED_r1 (Cascade|fixed)'
    description = 'Fixes the directions to Ball & Stick.'
    models = ('BallStick_r1 (Cascade)',
              'CHARMED_r1')
    inits = {'CHARMED_r1': [('Tensor.theta', 'Stick.theta'),
                            ('Tensor.phi', 'Stick.phi'),
                            ('w_res0.w', 'w_stick.w')]}
    fixes = {'CHARMED_r1': [('CHARMEDRestricted0.theta', 'Stick.theta'),
                            ('CHARMEDRestricted0.phi', 'Stick.phi')]}


class CHARMEDR2(CascadeConfig):

    name = 'CHARMED_r2 (Cascade)'
    description = 'Initializes the directions to 2x Ball & Stick.'
    models = ('BallStick_r2 (Cascade)',
              'CHARMED_r2')
    inits = {'CHARMED_r2': [('Tensor.theta', 'Stick0.theta'),
                            ('Tensor.phi', 'Stick0.phi'),
                            ('CHARMEDRestricted0.theta', 'Stick0.theta'),
                            ('CHARMEDRestricted0.phi', 'Stick0.phi'),
                            ('CHARMEDRestricted1.theta', 'Stick1.theta'),
                            ('CHARMEDRestricted1.phi', 'Stick1.phi'),
                            ('w_res0.w', 'w_stick0.w'),
                            ('w_res1.w', 'w_stick1.w')]}


class CHARMEDR2Fixed(CascadeConfig):

    name = 'CHARMED_r2 (Cascade|fixed)'
    description = 'Fixes the directions to 2x Ball & Stick.'
    models = ('BallStick_r2 (Cascade)',
              'CHARMED_r2')
    inits = {'CHARMED_r2': [('Tensor.theta', 'Stick0.theta'),
                            ('Tensor.phi', 'Stick0.phi'),
                            ('w_res0.w', 'w_stick0.w'),
                            ('w_res1.w', 'w_stick1.w')]}
    fixes = {'CHARMED_r2': [('CHARMEDRestricted0.theta', 'Stick0.theta'),
                            ('CHARMEDRestricted0.phi', 'Stick0.phi'),
                            ('CHARMEDRestricted1.theta', 'Stick1.theta'),
                            ('CHARMEDRestricted1.phi', 'Stick1.phi'),
                            ]}


class CHARMED_r3(CascadeConfig):

    name = 'CHARMED_r3 (Cascade)'
    description = 'Initializes the directions to 3x Ball & Stick.'
    models = ('BallStick_r3 (Cascade)',
              'CHARMED_r3')
    inits = {'CHARMED_r3': [('Tensor.theta', 'Stick0.theta'),
                            ('Tensor.phi', 'Stick0.phi'),
                            ('w_res0.w', 'w_stick0.w'),
                            ('w_res1.w', 'w_stick1.w'),
                            ('w_res2.w', 'w_stick2.w'),
                            ('CHARMEDRestricted0.theta', 'Stick0.theta'),
                            ('CHARMEDRestricted0.phi', 'Stick0.phi'),
                            ('CHARMEDRestricted1.theta', 'Stick1.theta'),
                            ('CHARMEDRestricted1.phi', 'Stick1.phi'),
                            ('CHARMEDRestricted2.theta', 'Stick2.theta'),
                            ('CHARMEDRestricted2.phi', 'Stick2.phi'),
                            ]}


class CHARMED_r3_Fixed(CascadeConfig):

    name = 'CHARMED_r3 (Cascade|fixed)'
    description = 'Fixes the directions to 3x Ball & Stick.'
    models = ('BallStick_r3 (Cascade)',
              'CHARMED_r3')
    inits = {'CHARMED_r3': [('Tensor.theta', 'Stick0.theta'),
                            ('Tensor.phi', 'Stick0.phi'),
                            ('w_res0.w', 'w_stick0.w'),
                            ('w_res1.w', 'w_stick1.w'),
                            ('w_res2.w', 'w_stick2.w')]}
    fixes = {'CHARMED_r3': [('CHARMEDRestricted0.theta', 'Stick0.theta'),
                            ('CHARMEDRestricted0.phi', 'Stick0.phi'),
                            ('CHARMEDRestricted1.theta', 'Stick1.theta'),
                            ('CHARMEDRestricted1.phi', 'Stick1.phi'),
                            ('CHARMEDRestricted2.theta', 'Stick2.theta'),
                            ('CHARMEDRestricted2.phi', 'Stick2.phi')]}
