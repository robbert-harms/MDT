import numpy as np

from mdt.models.cascade import CascadeConfig

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


class TensorS0(CascadeConfig):

    name = 'Tensor (Cascade|S0)'
    description = 'Cascade for Tensor initialized with only an S0 fit.'
    models = ('S0',
              'Tensor')


class TensorExVivo(Tensor):

    name = 'Tensor-ExVivo (Cascade)'
    description = 'Cascade for Tensor with ex vivo defaults.'
    models = ('BallStick_r1-ExVivo (Cascade)',
              'Tensor-ExVivo')


class TensorExVivoS0(Tensor):

    name = 'Tensor-ExVivo (Cascade|S0)'
    description = 'Cascade S0 for Tensor with ex vivo defaults.'
    models = ('S0',
              'Tensor-ExVivo')
