from mdt.components_config.cascade_models import CascadeConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tensor(CascadeConfig):

    name = 'Tensor (Cascade)'
    description = 'Cascade for Tensor.'
    models = ('BallStick_r1 (Cascade)',
              'Tensor')
    inits = {'Tensor': [('Tensor.theta', 'Stick.theta'),
                        ('Tensor.phi', 'Stick.phi')]}


class TensorFixed(CascadeConfig):

    name = 'Tensor (Cascade|fixed)'
    description = 'Cascade for Tensor with fixed angles.'
    models = ('BallStick_r1 (Cascade)',
              'Tensor')
    fixes = {'Tensor': [('Tensor.theta', 'Stick.theta'),
                        ('Tensor.phi', 'Stick.phi')]}
