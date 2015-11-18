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

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(Tensor, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'Tensor':
            model.init('Tensor.theta', output_previous['Stick.theta'])
            model.init('Tensor.phi', output_previous['Stick.phi'])


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