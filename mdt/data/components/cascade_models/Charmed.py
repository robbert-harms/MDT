import mdt
from mdt.cascade_model import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [get_charmed_r1(fixed=True),
            get_charmed_r1(fixed=False),

            {'model_constructor': Charmed2r,
             'name': Charmed2r_name,
             'description': 'Cascade for Charmed_2r initialized with two B&S.'},

            get_charmed(fixed=False),
            get_charmed(fixed=True)]



def get_charmed_r1(fixed=False):
    if fixed:
        name = 'Charmed_1r (Cascade|fixed)'
        description = 'Fixes the directions to Ball & Stick.'
    else:
        name = 'Charmed_1r (Cascade)'
        description = 'Initializes the directions to Ball & Stick.'

    def prepare_charmed(model, output_previous_model):
        if fixed:
            model.cmf('CharmedRestricted0')\
                .fix('theta', output_previous_model['Stick.theta'])\
                .fix('phi', output_previous_model['Stick.phi'])
        else:
            model.cmf('CharmedRestricted0')\
                .init('theta', output_previous_model['Stick.theta'])\
                .init('phi', output_previous_model['Stick.phi'])

        model.cmf('Tensor').init('theta', output_previous_model['Stick.theta'])\
                           .init('phi', output_previous_model['Stick.phi'])

    class Charmed_1r(SimpleCascadeModel):

        def __init__(self):
            super(Charmed_1r, self).__init__(
                name,
                (mdt.get_model('BallStick (Cascade)'),
                 mdt.get_model('Charmed_1r')))

        def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
            super(Charmed_1r, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
            if position == 1:
                prepare_charmed(model, output_previous_model)

    return {'model_constructor': Charmed_1r,
            'name': name,
            'description': description}


Charmed2r_name = 'Charmed_2r (Cascade|fixed)'
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
            model.cmf('Tensor').init('theta', output_previous_model['Stick0.theta'])\
                           .init('phi', output_previous_model['Stick0.phi'])

def get_charmed(fixed=False):
    if fixed:
        name = 'Charmed (Cascade|fixed)'
        description = 'Fixes the directions to Ball & 3 Sticks.'
    else:
        name = 'Charmed (Cascade)'
        description = 'Initializes the directions to Ball & 3 Sticks.'

    def prepare_charmed(model, output_previous_model):
        if fixed:
            for i in range(3):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous_model['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous_model['Stick' + repr(i) + '.phi'])
                model.cmf('w_res' + repr(i)).init('w', output_previous_model['Wstick' + repr(i) + '.w'])
        else:
            for i in range(3):
                model.cmf('CharmedRestricted' + repr(i))\
                    .init('theta', output_previous_model['Stick' + repr(i) + '.theta'])\
                    .init('phi', output_previous_model['Stick' + repr(i) + '.phi'])
                model.cmf('w_res' + repr(i)).init('w', output_previous_model['Wstick' + repr(i) + '.w'])

        model.cmf('Tensor').init('theta', output_previous_model['Stick0.theta'])\
                           .init('phi', output_previous_model['Stick0.phi'])

    class Charmed(SimpleCascadeModel):

        def __init__(self):
            super(Charmed, self).__init__(
                name,
                (mdt.get_model('BallStickStickStick (Cascade)'),
                 mdt.get_model('Charmed')))

        def _prepare_model(self, model, position, output_previous_model, output_all_previous_models):
            super(Charmed, self)._prepare_model(model, position, output_previous_model, output_all_previous_models)
            if position == 1:
                prepare_charmed(model, output_previous_model)

    return {'model_constructor': Charmed,
            'name': name,
            'description': description}