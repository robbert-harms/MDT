from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-07-19'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class AxCaliber(CascadeTemplate):

    models = ('BallStick_r1 (Cascade)',
              'AxCaliber')
    inits = {'AxCaliber': {'GDRCylinders.theta': 'Stick0.theta',
                           'GDRCylinders.phi': 'Stick0.phi'}}


class AxCaliber_Fixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    models = ('BallStick_r1 (Cascade)',
              'AxCaliber')
    fixes = {'AxCaliber': {'GDRCylinders.theta': 'Stick0.theta',
                           'GDRCylinders.phi': 'Stick0.phi'}}
