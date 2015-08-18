import mdt
from mdt.cascade_model import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [{'model_constructor': Charmed1r,
             'name': Charmed1r_name,
             'description': 'Cascade for Charmed_1r using only one B&S.'},

            {'model_constructor': Charmed2r,
             'name': Charmed2r_name,
             'description': 'Cascade for Charmed_2r initialized with two B&S.'},

            {'model_constructor': Charmed,
             'name': Charmed_name,
             'description': 'Cascade for Charmed using Ball and three Sticks as basis.'}]


Charmed1r_name = 'Charmed_1r (Cascade)'
class Charmed1r(SimpleCascadeModel):

    def __init__(self):
        super(Charmed1r, self).__init__(
            Charmed1r_name,
            (mdt.get_model('BallStick (Cascade)'),
             mdt.get_model('Charmed_1r')))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(Charmed1r, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 1:
            model.cmf('CharmedRestricted0')\
                .fix('theta', output_previous_model['Stick.theta'])\
                .fix('phi', output_previous_model['Stick.phi'])
            model.cmf('w_res0').init('w', output_previous_model['Wstick' + '.w'])


Charmed2r_name = 'Charmed_2r (Cascade)'
class Charmed2r(SimpleCascadeModel):

    def __init__(self):
        super(Charmed2r, self).__init__(
            Charmed2r_name,
            (mdt.get_model('BallStickStick (Cascade)'),
             mdt.get_model('Charmed_2r')))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(Charmed2r, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 1:
            for i in range(2):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous_model['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous_model['Stick' + repr(i) + '.phi'])
                model.cmf('w_res' + repr(i)).init('w', output_previous_model['Wstick' + repr(i) + '.w'])


Charmed_name = 'Charmed (Cascade)'
class Charmed(SimpleCascadeModel):

    def __init__(self):
        super(Charmed, self).__init__(
            Charmed_name,
            (mdt.get_model('BallStickStickStick (Cascade)'),
             mdt.get_model('Charmed')))

    def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
        super(Charmed, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
        if position == 1:
            for i in range(3):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous_model['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous_model['Stick' + repr(i) + '.phi'])
                model.cmf('w_res' + repr(i)).init('w', output_previous_model['Wstick' + repr(i) + '.w'])
            model.cmf('Tensor').init('theta', output_previous_model['Stick0.theta'])\
                               .init('phi', output_previous_model['Stick0.phi'])