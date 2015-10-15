from mdt.cascade_model import cascade_builder_decorator, SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [Tensor.get_meta_data(),
            TensorExVivo.get_meta_data(),
            TensorT2.get_meta_data()
            ]


@cascade_builder_decorator
class Tensor(SimpleCascadeModel):

    name = 'Tensor (Cascade)'
    description = 'Cascade for Tensor.'
    models = ('BallStick (Cascade)',
              'Tensor')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(Tensor, self)._prepare_model(model, position, output_previous, output_all_previous)
        if position == 1:
            model.init('Tensor.theta', output_previous['Stick.theta'])
            model.init('Tensor.phi', output_previous['Stick.phi'])


@cascade_builder_decorator
class TensorExVivo(Tensor):

    name = 'Tensor-ExVivo (Cascade)'
    description = 'Cascade for Tensor with ex vivo defaults.'
    models = ('BallStick-ExVivo (Cascade)',
              'Tensor-ExVivo')


@cascade_builder_decorator
class TensorT2(Tensor):

    name = 'Tensor-T2 (Cascade)'
    description = 'Cascade for Tensor with an extra T2 model.'
    models = ('BallStick-T2 (Cascade)',
              'Tensor-t2')