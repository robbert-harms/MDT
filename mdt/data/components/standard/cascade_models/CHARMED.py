from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CHARMED_r1(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'CHARMED_r1')
    inits = {'CHARMED_r1': {'CHARMEDRestricted0.theta': 'Stick0.theta',
                            'CHARMEDRestricted0.phi': 'Stick0.phi',
                            'Tensor.theta': 'Stick0.theta',
                            'Tensor.phi': 'Stick0.phi',
                            'w_res0.w': 'w_stick0.w'}}


class CHARMED_r1_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'CHARMED_r1')
    inits = {'CHARMED_r1': {'Tensor.theta': 'Stick0.theta',
                            'Tensor.phi': 'Stick0.phi',
                            'w_res0.w': 'w_stick0.w'}}
    fixes = {'CHARMED_r1': {'CHARMEDRestricted0.theta': 'Stick0.theta',
                            'CHARMEDRestricted0.phi': 'Stick0.phi'}}


class CHARMED_r2(CascadeTemplate):

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


class CHARMED_r2_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
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


class CHARMED_r3(CascadeTemplate):

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


class CHARMED_r3_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
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
