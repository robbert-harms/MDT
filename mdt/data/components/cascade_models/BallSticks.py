import mdt
from mdt.cascade_model import SimpleCascadeBuilder, SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [#BallStick().build(),
            BallStick.get_meta_data(),
            BallStickExVivo().build(),
            BallStickStick().build(),
            BallStickStickExVivo().build(),
            BallStickStickStick().build(),
            BallStickStickStickExVivo().build()]





class BallStick(SimpleCascadeModel):

    def __init__(self):
        cascade_names = ('s0', 'BallStick')
        cascade = list(map(mdt.get_model, cascade_names))
        super(BallStick, self).__init__(BallStick.get_meta_data()['name'], cascade)

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(BallStick, self)._prepare_model(model, position, output_previous, output_all_previous)

    @staticmethod
    def get_meta_data():
        return {
            'model_constructor': BallStick,
            'name': 'BallStick (Cascade)',
            'description': 'Cascade for Ballstick'
        }

# class BallStick(SimpleCascadeBuilder):
#
#     def _get_name(self):
#         return 'BallStick (Cascade)'
#
#     def _get_description(self):
#         return 'Cascade for BallStick.'
#
#     def _get_cascade_names(self):
#         return ('s0',
#                 'BallStick')


class BallStickExVivo(SimpleCascadeBuilder):

    def _get_name(self):
        return 'BallStick-ExVivo (Cascade)'

    def _get_description(self):
        return 'Cascade for BallStick with ex vivo defaults.'

    def _get_cascade_names(self):
        return ('s0',
                'BallStick-ExVivo')


class BallStickStick(SimpleCascadeBuilder):

    def _get_name(self):
        return 'BallStickStick (Cascade)'

    def _get_description(self):
        return 'Cascade for BallStickStick.'

    def _get_cascade_names(self):
        return ('BallStick (Cascade)',
                'BallStickStick')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                model.init('Stick0.theta', output_previous['Stick.theta'])
                model.init('Stick0.phi', output_previous['Stick.phi'])
                model.init('Wstick0.w', output_previous['Wstick.w'])
                model.init('Wstick1.w', 0.0)
        return _prepare_model


class BallStickStickExVivo(BallStickStick):

    def _get_name(self):
        return 'BallStickStick-ExVivo (Cascade)'

    def _get_description(self):
        return 'Cascade for BallStickStick with ex vivo defaults.'

    def _get_cascade_names(self):
        return ('BallStick-ExVivo (Cascade)',
                'BallStickStick-ExVivo')


class BallStickStickStick(SimpleCascadeBuilder):

    def _get_name(self):
        return 'BallStickStickStick (Cascade)'

    def _get_description(self):
        return 'Cascade for BallStickStickStick.'

    def _get_cascade_names(self):
        return ('BallStickStick (Cascade)',
                'BallStickStickStick')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 1:
                model.fix('Stick0.theta', output_previous['Stick0.theta'])
                model.fix('Stick0.phi', output_previous['Stick0.phi'])
                model.init('Wstick1.w', 0.0)
        return _prepare_model


class BallStickStickStickExVivo(BallStickStick):

    def _get_name(self):
        return 'BallStickStickStick-ExVivo (Cascade)'

    def _get_description(self):
        return 'Cascade for BallStickStickStick with ex vivo defaults.'

    def _get_cascade_names(self):
        return ('BallStickStick-ExVivo (Cascade)',
                'BallStickStickStick-ExVivo')