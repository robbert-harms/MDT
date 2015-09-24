import math
import mdt
from mdt.cascade_model import SimpleCascadeModel
from mot.evaluation_models import OffsetGaussianEvaluationModel
from mot.signal_noise_models import JohnsonSignalNoise

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [{'model_constructor': NoddiInit,
             'name': Noddi_JN_init_name,
             'description': 'Cascade for Noddi with JohnsonNoise as signal noise.'},
            {'model_constructor': NoddiGONInit,
             'name': Noddi_GON_init_name,
             'description': 'Cascade for Noddi with OffSetGaussian noise as evaluation model.'},
            {'model_constructor': NoddiGONJNInit,
             'name': Noddi_GON_JN_init_name,
             'description': 'Cascade for Noddi with OffSetGaussian noise as evaluation model and with JohnsonNoise.'},]


Noddi_JN_init_name = 'Noddi (Cascade|JN)'
class NoddiInit(SimpleCascadeModel):

    def __init__(self):
        super(NoddiInit, self).__init__(
            Noddi_JN_init_name,
            (mdt.get_model('s0'),
             mdt.get_model('BallStick'),
             mdt.get_model('Noddi', signal_noise_model=JohnsonSignalNoise()),))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(NoddiInit, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 2:
            model.cmf('Wic').init('w', output_previous_model['Wstick.w']/2.0)
            model.cmf('Wec').init('w', output_previous_model['Wstick.w']/2.0)
            model.cmf('Wcsf').init('w', output_previous_model['Wball.w'])
            model.cmf('Noddi_IC').init('theta', output_previous_model['Stick.theta'])
            model.cmf('Noddi_IC').init('phi', output_previous_model['Stick.phi'])


Noddi_GON_init_name = 'Noddi (Cascade|GON)'
class NoddiGONInit(SimpleCascadeModel):

    def __init__(self):
        super(NoddiGONInit, self).__init__(
            Noddi_GON_init_name,
            (mdt.get_model('s0'),
             mdt.get_model('BallStick'),
             mdt.get_model('Noddi', evaluation_model=OffsetGaussianEvaluationModel()),))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(NoddiGONInit, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 2:
            model.cmf('Wic').init('w', output_previous_model['Wstick.w']/2.0)
            model.cmf('Wec').init('w', output_previous_model['Wstick.w']/2.0)
            model.cmf('Wcsf').init('w', output_previous_model['Wball.w'])
            model.cmf('Noddi_IC').init('theta', output_previous_model['Stick.theta'])
            model.cmf('Noddi_IC').init('phi', output_previous_model['Stick.phi'])


Noddi_GON_JN_init_name = 'Noddi (Cascade|GON|JN)'
class NoddiGONJNInit(SimpleCascadeModel):

    def __init__(self):
        super(NoddiGONJNInit, self).__init__(
            Noddi_GON_JN_init_name,
            (mdt.get_model('s0'),
             mdt.get_model('BallStick'),
             mdt.get_model('Noddi', evaluation_model=OffsetGaussianEvaluationModel(),
                           signal_noise_model=JohnsonSignalNoise()),))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(NoddiGONJNInit, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 2:
            model.cmf('Wic').init('w', output_previous_model['Wstick.w']/2.0)
            model.cmf('Wec').init('w', output_previous_model['Wstick.w']/2.0)
            model.cmf('Wcsf').init('w', output_previous_model['Wball.w'])
            model.cmf('Noddi_IC').init('theta', output_previous_model['Stick.theta'])
            model.cmf('Noddi_IC').init('phi', output_previous_model['Stick.phi'])