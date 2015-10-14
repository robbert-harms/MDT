from mdt.cascade_model import SimpleCascadeBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [CharmedR1().build(),
            CharmedR1_fixed().build(),
            CharmedR2().build(),
            CharmedR2_fixed().build(),
            Charmed().build(),
            Charmed_fixed().build()]


class CharmedR1(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Charmed_1r (Cascade)'

    def _get_description(self):
        return 'Initializes the directions to Ball & Stick.'

    def _get_cascade_names(self):
        return ('BallStick (Cascade)',
                'Charmed_1r')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                model.cmf('CharmedRestricted0')\
                    .init('theta', output_previous['Stick.theta'])\
                    .init('phi', output_previous['Stick.phi'])

                model.cmf('Tensor')\
                    .init('theta', output_previous['Stick.theta'])\
                    .init('phi', output_previous['Stick.phi'])

        return _prepare_model


class CharmedR1_fixed(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Charmed_1r (Cascade|fixed)'

    def _get_description(self):
        return 'Fixes the directions to Ball & Stick.'

    def _get_cascade_names(self):
        return ('BallStick (Cascade)',
                'Charmed_1r')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                model.cmf('CharmedRestricted0')\
                    .fix('theta', output_previous['Stick.theta'])\
                    .fix('phi', output_previous['Stick.phi'])

                model.cmf('Tensor')\
                    .init('theta', output_previous['Stick.theta'])\
                    .init('phi', output_previous['Stick.phi'])

        return _prepare_model


class CharmedR2(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Charmed_2r (Cascade)'

    def _get_description(self):
        return 'Initializes the directions to 2x Ball & Stick.'

    def _get_cascade_names(self):
        return ('BallStickStick (Cascade)',
                'Charmed_2r')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                for i in range(2):
                    model.cmf('CharmedRestricted' + repr(i))\
                        .init('theta', output_previous['Stick' + repr(i) + '.theta'])\
                        .init('phi', output_previous['Stick' + repr(i) + '.phi'])

                    model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

                model.cmf('Tensor')\
                    .init('theta', output_previous['Stick0.theta'])\
                    .init('phi', output_previous['Stick0.phi'])

        return _prepare_model


class CharmedR2_fixed(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Charmed_2r (Cascade|fixed)'

    def _get_description(self):
        return 'Fixes the directions to 2x Ball & Stick.'

    def _get_cascade_names(self):
        return ('BallStickStick (Cascade)',
                'Charmed_2r')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                for i in range(2):
                    model.cmf('CharmedRestricted' + repr(i))\
                        .fix('theta', output_previous['Stick' + repr(i) + '.theta'])\
                        .fix('phi', output_previous['Stick' + repr(i) + '.phi'])

                    model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

                model.cmf('Tensor')\
                    .init('theta', output_previous['Stick0.theta'])\
                    .init('phi', output_previous['Stick0.phi'])

        return _prepare_model


class Charmed(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Charmed (Cascade)'

    def _get_description(self):
        return 'Initializes the directions to 3x Ball & Stick.'

    def _get_cascade_names(self):
        return ('BallStickStickStick (Cascade)',
                'Charmed')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                for i in range(3):
                    model.cmf('CharmedRestricted' + repr(i))\
                        .init('theta', output_previous['Stick' + repr(i) + '.theta'])\
                        .init('phi', output_previous['Stick' + repr(i) + '.phi'])

                    model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

                model.cmf('Tensor')\
                    .init('theta', output_previous['Stick0.theta'])\
                    .init('phi', output_previous['Stick0.phi'])

        return _prepare_model


class Charmed_fixed(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Charmed (Cascade|fixed)'

    def _get_description(self):
        return 'Fixes the directions to 3x Ball & Stick.'

    def _get_cascade_names(self):
        return ('BallStickStickStick (Cascade)',
                'Charmed')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                for i in range(3):
                    model.cmf('CharmedRestricted' + repr(i))\
                        .fix('theta', output_previous['Stick' + repr(i) + '.theta'])\
                        .fix('phi', output_previous['Stick' + repr(i) + '.phi'])

                    model.cmf('w_res' + repr(i)).init('w', output_previous['Wstick' + repr(i) + '.w'])

                model.cmf('Tensor')\
                    .init('theta', output_previous['Stick0.theta'])\
                    .init('phi', output_previous['Stick0.phi'])

        return _prepare_model