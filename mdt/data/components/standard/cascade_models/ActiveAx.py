from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-07-19'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ActiveAx_ExVivo(CascadeTemplate):

    description = 'Initializes the directions to Ball & Stick.'
    models = ('BallStick_r1 (Cascade)',
              'ActiveAx_ExVivo')
    inits = {'ActiveAx_ExVivo': {'CylinderGPD.theta': 'Stick0.theta',
                                 'CylinderGPD.phi': 'Stick0.phi'}}


class ActiveAx(CascadeTemplate):

    description = 'Initializes the directions to Ball & Stick.'
    models = ('BallStick_r1 (Cascade)',
              'ActiveAx')
    inits = {'ActiveAx': {'CylinderGPD.theta': 'Stick0.theta',
                          'CylinderGPD.phi': 'Stick0.phi'}}
