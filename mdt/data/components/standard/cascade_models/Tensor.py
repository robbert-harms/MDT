from mdt.models.cascade import CascadeModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Tensor(CascadeModelBuilder):

    name = 'Tensor (Cascade)'
    description = 'Cascade for Tensor.'
    models = ('BallStick (Cascade)',
              'Tensor')
    inits = {'Tensor': [('Tensor.theta', 'Stick.theta'),
                        ('Tensor.phi', 'Stick.phi')]}


class TensorFixed(CascadeModelBuilder):

    name = 'Tensor (Cascade|fixed)'
    description = 'Cascade for Tensor with fixed angles.'
    models = ('BallStick (Cascade)',
              'Tensor')
    fixes = {'Tensor': [('Tensor.theta', 'Stick.theta'),
                        ('Tensor.phi', 'Stick.phi')]}


class Tensors0(CascadeModelBuilder):

    name = 'Tensor (Cascade|s0)'
    description = 'Cascade for Tensor initialized with only an S0 fit.'
    models = ('s0',
              'Tensor')


class TensorExVivo(Tensor):

    name = 'Tensor-ExVivo (Cascade)'
    description = 'Cascade for Tensor with ex vivo defaults.'
    models = ('BallStick-ExVivo (Cascade)',
              'Tensor-ExVivo')


class TensorT2(Tensor):

    name = 'Tensor-T2 (Cascade)'
    description = 'Cascade for Tensor with an extra T2 model.'
    models = ('BallStick-T2 (Cascade)',
              'Tensor-t2')