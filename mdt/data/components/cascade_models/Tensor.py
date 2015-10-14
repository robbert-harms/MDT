from mdt.cascade_model import SimpleCascadeBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [Tensor().build(),
            TensorExVivo().build(),
            TensorT2().build()
            ]


class Tensor(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Tensor (Cascade)'

    def _get_description(self):
        return 'Cascade for Tensor.'

    def _get_cascade_names(self):
        return ('BallStick (Cascade)',
                'Tensor')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                model.init('Tensor.theta', output_previous['Stick.theta'])
                model.init('Tensor.phi', output_previous['Stick.phi'])
        return _prepare_model


class TensorExVivo(Tensor):

    def _get_name(self):
        return 'Tensor-ExVivo (Cascade)'

    def _get_description(self):
        return 'Cascade for Tensor with ex vivo defaults.'

    def _get_cascade_names(self):
        return ('BallStick-ExVivo (Cascade)',
                'Tensor-ExVivo')


class TensorT2(Tensor):

    def _get_name(self):
        return 'Tensor-T2 (Cascade)'

    def _get_description(self):
        return 'Cascade for Tensor with an extra T2 model.'

    def _get_cascade_names(self):
        return ('BallStick-T2 (Cascade)',
                'Tensor-T2')