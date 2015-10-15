from mdt.cascade_model import SimpleCascadeModel, cascade_builder_decorator

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [CharmedR1.get_meta_data(),
            CharmedR1Fixed.get_meta_data(),
            CharmedR2.get_meta_data(),
            CharmedR2Fixed.get_meta_data(),
            Charmed.get_meta_data(),
            CharmedFixed.get_meta_data()]


@cascade_builder_decorator
class CharmedR1(SimpleCascadeModel):

    name = 'Charmed_1r (Cascade)'
    description = 'Initializes the directions to Ball & Stick.'
    models = ('BallStick (Cascade)',
              'Charmed_1r')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(CharmedR1, self)._prepare_model(model, position, output_previous, output_all_previous)
        if position == 1:
            model.cmf('CharmedRestricted0')\
                .init('theta', output_previous['Stick.theta'])\
                .init('phi', output_previous['Stick.phi'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick.theta'])\
                .init('phi', output_previous['Stick.phi'])


@cascade_builder_decorator
class CharmedR1Fixed(SimpleCascadeModel):

    name = 'Charmed_1r (Cascade|fixed)'
    description = 'Fixes the directions to Ball & Stick.'
    models = ('BallStick (Cascade)',
              'Charmed_1r')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(CharmedR1Fixed, self)._prepare_model(model, position, output_previous, output_all_previous)
        if position == 1:
            model.cmf('CharmedRestricted0')\
                .fix('theta', output_previous['Stick.theta'])\
                .fix('phi', output_previous['Stick.phi'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick.theta'])\
                .init('phi', output_previous['Stick.phi'])


@cascade_builder_decorator
class CharmedR2(SimpleCascadeModel):

    name = 'Charmed_2r (Cascade)'
    description = 'Initializes the directions to 2x Ball & Stick.'
    models = ('BallStickStick (Cascade)',
              'Charmed_2r')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(CharmedR2, self)._prepare_model(model, position, output_previous, output_all_previous)

        if position == 1:
            for i in range(2):
                model.cmf('CharmedRestricted' + repr(i))\
                    .init('theta', output_previous['Stick' + repr(i) + '.theta'])\
                    .init('phi', output_previous['Stick' + repr(i) + '.phi'])

                model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick0.theta'])\
                .init('phi', output_previous['Stick0.phi'])


@cascade_builder_decorator
class CharmedR2Fixed(SimpleCascadeModel):

    name = 'Charmed_2r (Cascade|fixed)'
    description = 'Fixes the directions to 2x Ball & Stick.'
    models = ('BallStickStick (Cascade)',
              'Charmed_2r')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(CharmedR2Fixed, self)._prepare_model(model, position, output_previous, output_all_previous)

        if position == 1:
            for i in range(2):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous['Stick' + repr(i) + '.phi'])

                model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick0.theta'])\
                .init('phi', output_previous['Stick0.phi'])


@cascade_builder_decorator
class Charmed(SimpleCascadeModel):

    name = 'Charmed (Cascade)'
    description = 'Initializes the directions to 3x Ball & Stick.'
    models = ('BallStickStickStick (Cascade)',
              'Charmed')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(Charmed, self)._prepare_model(model, position, output_previous, output_all_previous)

        if position == 1:
            for i in range(3):
                model.cmf('CharmedRestricted' + repr(i))\
                    .init('theta', output_previous['Stick' + repr(i) + '.theta'])\
                    .init('phi', output_previous['Stick' + repr(i) + '.phi'])

                model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick0.theta'])\
                .init('phi', output_previous['Stick0.phi'])


@cascade_builder_decorator
class CharmedFixed(SimpleCascadeModel):

    name = 'Charmed (Cascade|fixed)'
    description = 'Fixes the directions to 3x Ball & Stick.'
    models = ('BallStickStickStick (Cascade)',
              'Charmed')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(CharmedFixed, self)._prepare_model(model, position, output_previous, output_all_previous)

        if position == 1:
            for i in range(3):
                model.cmf('CharmedRestricted' + repr(i))\
                    .fix('theta', output_previous['Stick' + repr(i) + '.theta'])\
                    .fix('phi', output_previous['Stick' + repr(i) + '.phi'])

                model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

            model.cmf('Tensor')\
                .init('theta', output_previous['Stick0.theta'])\
                .init('phi', output_previous['Stick0.phi'])