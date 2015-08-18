import mdt
from mdt.cascade_model import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [{'model_constructor': Noddi,
             'name': Noddi_name,
             'description': 'Cascade for Noddi'}]


Noddi_name = 'Noddi (Cascade)'
class Noddi(SimpleCascadeModel):

    def __init__(self):
        super(Noddi, self).__init__(
            Noddi_name,
            (mdt.get_model('s0'),
             mdt.get_model('BallStick'),
             mdt.get_model('Noddi'),))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(Noddi, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 2:
            model.cmf('Wic').init('w', output_previous_model['Wstick.w'])
            model.cmf('Wec').init('w', output_previous_model['Wball.w'])
            model.cmf('Noddi_IC').init('theta', output_previous_model['Stick.theta'])
            model.cmf('Noddi_IC').init('phi', output_previous_model['Stick.phi'])