import mdt
from mdt.cascade_model import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [{'model_constructor': TensorT2,
             'name': TensorT2_name,
             'description': 'Cascade for Tensor with one T2.'}]


TensorT2_name = 'Tensor-T2 (Cascade)'
class TensorT2(SimpleCascadeModel):

    def __init__(self):
        super(TensorT2, self).__init__(
            TensorT2_name,
            (mdt.get_model('s0'),
             mdt.get_model('s0-T2'),
             mdt.get_model('BallStick-T2'),
             mdt.get_model('Tensor-T2'),))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(TensorT2, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 3:
            model.cmf('Tensor').init('theta', output_previous_model['Stick.theta'])
            model.cmf('Tensor').init('phi', output_previous_model['Stick.phi'])