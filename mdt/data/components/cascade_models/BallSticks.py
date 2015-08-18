import mdt
from mdt.cascade_model import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [{'model_constructor': BallStick,
             'name': BallStick_name,
             'description': 'Cascade for BallStick.'},

            {'model_constructor': BallStickExVivo,
             'name': BallStickExVivo_name,
             'description': 'Cascade for BallStick ex vivo.'},

            {'model_constructor': BallStickStick,
             'name': BallStickStick_name,
             'description': 'Cascade for Ball and 2 Sticks.'},

            {'model_constructor': BallStickStickStick,
             'name': BallStickStickStick_name,
             'description': 'Cascade for Ball and 3 Sticks.'}]


BallStick_name = 'BallStick (Cascade)'
class BallStick(SimpleCascadeModel):

    def __init__(self):
        super(BallStick, self).__init__(
            BallStick_name,
            (mdt.get_model('s0'),
             mdt.get_model('BallStick'),))


BallStickExVivo_name = 'BallStick-ExVivo (Cascade)'
class BallStickExVivo(SimpleCascadeModel):

    def __init__(self):
        super(BallStickExVivo, self).__init__(
            BallStickExVivo_name,
            (mdt.get_model('s0'),
             mdt.get_model('BallStick-ExVivo'),))


BallStickStick_name = 'BallStickStick (Cascade)'
class BallStickStick(SimpleCascadeModel):

    def __init__(self):
        super(BallStickStick, self).__init__(
            BallStickStick_name,
            (BallStick(),
             mdt.get_model('BallStickStick')))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(BallStickStick, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 1:
            model.cmf('Stick0').init('theta', output_previous_model['Stick.theta'])
            model.cmf('Stick0').init('phi', output_previous_model['Stick.phi'])
            model.cmf('Wstick0').init('w', output_previous_model['Wstick.w'])
            model.cmf('Wstick1').init('w', 0.0)


BallStickStickStick_name = 'BallStickStickStick (Cascade)'
class BallStickStickStick(SimpleCascadeModel):

    def __init__(self):
        super(BallStickStickStick, self).__init__(
            BallStickStickStick_name,
            (BallStickStick(),
             mdt.get_model('BallStickStickStick'),))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(BallStickStickStick, self)._prepare_model(model, position,
                                                        output_previous_model, output_all_previous_models)
        if position == 1:
            model.cmf('Stick0').fix('theta', output_previous_model['Stick0.theta'])
            model.cmf('Stick0').fix('phi', output_previous_model['Stick0.phi'])
            model.cmf('Wstick1').init('w', 0.0)