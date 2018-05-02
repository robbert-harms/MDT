from mdt import CascadeTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tensor(CascadeTemplate):

    description = 'Cascade for Tensor.'
    models = ('BallStick_r1 (Cascade)',
              'Tensor')
    inits = {'Tensor': {'Tensor.theta': 'Stick0.theta',
                        'Tensor.phi': 'Stick0.phi'}}


class TensorFixed(CascadeTemplate):

    cascade_name_modifier = 'fixed'
    description = 'Cascade for Tensor with fixed angles.'
    models = ('BallStick_r1 (Cascade)',
              'Tensor')
    fixes = {'Tensor': {'Tensor.theta': 'Stick0.theta',
                        'Tensor.phi': 'Stick0.phi'}}
