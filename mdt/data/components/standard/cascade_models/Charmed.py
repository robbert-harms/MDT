from mdt.models.cascade import CascadeModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CharmedR1(CascadeModelBuilder):

    name = 'Charmed_1r (Cascade)'
    description = 'Initializes the directions to Ball & Stick.'
    models = ('BallStick (Cascade)',
              'Charmed_1r')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(CharmedR1, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'Charmed_1r':
            model.cmf('CharmedRestricted0')\
                .init('theta', output_previous['Stick.theta'])\
                .init('phi', output_previous['Stick.phi'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick.theta'])\
                .init('phi', output_previous['Stick.phi'])


class CharmedR1Fixed(CascadeModelBuilder):

    name = 'Charmed_1r (Cascade|fixed)'
    description = 'Fixes the directions to Ball & Stick.'
    models = ('BallStick (Cascade)',
              'Charmed_1r')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(CharmedR1Fixed, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'Charmed_1r':
            model.cmf('CharmedRestricted0')\
                .fix('theta', output_previous['Stick.theta'])\
                .fix('phi', output_previous['Stick.phi'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick.theta'])\
                .init('phi', output_previous['Stick.phi'])


class CharmedR2(CascadeModelBuilder):

    name = 'Charmed_2r (Cascade)'
    description = 'Initializes the directions to 2x Ball & Stick.'
    models = ('BallStickStick (Cascade)',
              'Charmed_2r')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(CharmedR2, self)._prepare_model(model, output_previous, output_all_previous)

        if model.name == 'Charmed_2r':
            for i in range(2):
                model.cmf('CharmedRestricted' + repr(i))\
                    .init('theta', output_previous['Stick' + repr(i) + '.theta'])\
                    .init('phi', output_previous['Stick' + repr(i) + '.phi'])

                model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick0.theta'])\
                .init('phi', output_previous['Stick0.phi'])


class CharmedR2Fixed(CascadeModelBuilder):

    name = 'Charmed_2r (Cascade|fixed)'
    description = 'Fixes the directions to 2x Ball & Stick.'
    models = ('BallStickStick (Cascade)',
              'Charmed_2r')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(CharmedR2Fixed, self)._prepare_model(model, output_previous, output_all_previous)

        if model.name == 'Charmed_2r':
            for i in range(2):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous['Stick' + repr(i) + '.phi'])

                model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick0.theta'])\
                .init('phi', output_previous['Stick0.phi'])


class Charmed(CascadeModelBuilder):

    name = 'Charmed (Cascade)'
    description = 'Initializes the directions to 3x Ball & Stick.'
    models = ('BallStickStickStick (Cascade)',
              'Charmed')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(Charmed, self)._prepare_model(model, output_previous, output_all_previous)

        if model.name == 'Charmed':
            for i in range(3):
                model.cmf('CharmedRestricted' + repr(i))\
                    .init('theta', output_previous['Stick' + repr(i) + '.theta'])\
                    .init('phi', output_previous['Stick' + repr(i) + '.phi'])

                model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick0.theta'])\
                .init('phi', output_previous['Stick0.phi'])


class CharmedFixed(CascadeModelBuilder):

    name = 'Charmed (Cascade|fixed)'
    description = 'Fixes the directions to 3x Ball & Stick.'
    models = ('BallStickStickStick (Cascade)',
              'Charmed')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(CharmedFixed, self)._prepare_model(model, output_previous, output_all_previous)

        if model.name == 'Charmed':
            for i in range(3):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous['Stick' + repr(i) + '.phi'])

                model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick0.theta'])\
                .init('phi', output_previous['Stick0.phi'])