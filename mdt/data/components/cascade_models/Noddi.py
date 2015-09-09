import mdt
from mdt.cascade_model import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [{'model_constructor': NoddiInit,
             'name': Noddi_init_name,
             'description': 'Cascade for Noddi with initialized directions.'},
            {'model_constructor': NoddiFixed,
             'name': Noddi_fixed_name,
             'description': 'Cascade for Noddi with fixed directions.'}]


Noddi_init_name = 'Noddi (Cascade)'
class NoddiInit(SimpleCascadeModel):

    def __init__(self):
        super(NoddiInit, self).__init__(
            Noddi_init_name,
            (mdt.get_model('s0'),
             mdt.get_model('BallStick'),
             mdt.get_model('Noddi'),))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(NoddiInit, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 2:
            model.cmf('Wic').init('w', output_previous_model['Wstick.w'])
            model.cmf('Wec').init('w', output_previous_model['Wball.w'])
            model.cmf('Noddi_IC').init('theta', output_previous_model['Stick.theta'])
            model.cmf('Noddi_IC').init('phi', output_previous_model['Stick.phi'])


Noddi_fixed_name = 'Noddi (Cascade|fixed)'
class NoddiFixed(SimpleCascadeModel):

    def __init__(self):
        super(NoddiFixed, self).__init__(
            Noddi_fixed_name,
            (mdt.get_model('s0'),
             mdt.get_model('BallStick'),
             mdt.get_model('Noddi'),))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(NoddiFixed, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 2:
            model.cmf('Wic').init('w', output_previous_model['Wstick.w'])
            model.cmf('Wec').init('w', output_previous_model['Wball.w'])
            model.cmf('Noddi_IC').fix('theta', output_previous_model['Stick.theta'])
            model.cmf('Noddi_IC').fix('phi', output_previous_model['Stick.phi'])