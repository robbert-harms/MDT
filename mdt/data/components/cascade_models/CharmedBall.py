import mdt
from mdt.cascade_model import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [{'model_constructor': CharmedBall1r,
             'name': CharmedBall1r_name,
             'description': 'Cascade for CharmedBall_1r using only one B&S.'},

            {'model_constructor': CharmedBall2r,
             'name': CharmedBall2r_name,
             'description': 'Cascade for CharmedBall_2r initialized with two B&S.'},

            {'model_constructor': CharmedBall,
             'name': CharmedBallname,
             'description': 'Cascade for CharmedBall initialized with Ball and three Sticks.'}]


CharmedBall1r_name = 'CharmedBall_1r (Cascade)'
class CharmedBall1r(SimpleCascadeModel):

    def __init__(self):
        super(CharmedBall1r, self).__init__(
            CharmedBall1r_name,
            (mdt.get_model('BallStick (Cascade)'),
             mdt.get_model('CharmedBall_1r')))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(CharmedBall1r, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 1:
            model.cmf('CharmedRestricted0')\
                .fix('theta', output_previous_model['Stick.theta'])\
                .fix('phi', output_previous_model['Stick.phi'])
            model.cmf('w_res0').init('w', output_previous_model['Wstick' + '.w'])


CharmedBall2r_name = 'CharmedBall_2r (Cascade)'
class CharmedBall2r(SimpleCascadeModel):

    def __init__(self):
        super(CharmedBall2r, self).__init__(
            CharmedBall2r_name,
            (mdt.get_model('BallStickStick (Cascade)'),
             mdt.get_model('CharmedBall_2r')))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(CharmedBall2r, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 1:
            for i in range(2):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous_model['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous_model['Stick' + repr(i) + '.phi'])
                model.cmf('w_res' + repr(i)).init('w', output_previous_model['Wstick' + repr(i) + '.w'])


CharmedBallname = 'CharmedBall (Cascade)'
class CharmedBall(SimpleCascadeModel):

    def __init__(self):
        super(CharmedBall, self).__init__(
            CharmedBallname,
            (mdt.get_model('BallStickStickStick (Cascade)'),
             mdt.get_model('CharmedBall')))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(CharmedBall, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 1:
            for i in range(3):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous_model['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous_model['Stick' + repr(i) + '.phi'])
                model.cmf('w_res' + repr(i)).init('w', output_previous_model['Wstick' + repr(i) + '.w'])
            model.cmf('Tensor').init('theta', output_previous_model['Stick0.theta'])\
                               .init('phi', output_previous_model['Stick0.phi'])