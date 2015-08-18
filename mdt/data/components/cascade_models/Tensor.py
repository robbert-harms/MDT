import mdt
from mdt.cascade_model import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [{'model_constructor': Tensor,
             'name': Tensor_name,
             'description': 'Cascade for Tensor.'},

            {'model_constructor': TensorExVivo,
             'name': TensorExVivo_name,
             'description': 'Cascade for Tensor with ex vivo defaults.'},
            ]


Tensor_name = 'Tensor (Cascade)'
class Tensor(SimpleCascadeModel):

    def __init__(self, name=None, cascade_models=None):
        name = name or Tensor_name
        cascade_models = cascade_models or (mdt.get_model('s0'),
                                            mdt.get_model('BallStick'),
                                            mdt.get_model('Tensor'),)
        super(Tensor, self).__init__(name, cascade_models)

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(Tensor, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 2:
            model.cmf('Tensor').init('theta', output_previous_model['Stick.theta'])
            model.cmf('Tensor').init('phi', output_previous_model['Stick.phi'])


TensorExVivo_name = 'Tensor-ExVivo (Cascade)'
class TensorExVivo(Tensor):

    def __init__(self):
        super(TensorExVivo, self).__init__(name = TensorExVivo_name,
                                           cascade_models=(mdt.get_model('s0'),
                                                           mdt.get_model('BallStick-ExVivo'),
                                                           mdt.get_model('Tensor-ExVivo'),))