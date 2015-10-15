import mdt
from mdt.cascade_model import SimpleCascadeModel, cascade_builder_decorator

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [BallStick.get_meta_data(),
            BallStickExVivo.get_meta_data(),
            BallStickStick.get_meta_data(),
            BallStickStickExVivo.get_meta_data(),
            BallStickStickStick.get_meta_data(),
            BallStickStickStickExVivo.get_meta_data()]


@cascade_builder_decorator
class BallStick(SimpleCascadeModel):

    name = 'BallStick (Cascade)'
    description = 'Cascade for Ballstick'
    models = ('s0',
              'BallStick')


@cascade_builder_decorator
class BallStickExVivo(BallStick):

    name = 'BallStick-ExVivo (Cascade)'
    description = 'Cascade for Ballstick with ex vivo defaults.'
    models = ('s0',
              'BallStick-ExVivo')


@cascade_builder_decorator
class BallStickStick(SimpleCascadeModel):

    name = 'BallStickStick (Cascade)'
    description = 'Cascade for BallStickStick.'
    models = ('BallStick (Cascade)',
              'BallStickStick')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(BallStickStick, self)._prepare_model(model, position, output_previous, output_all_previous)
        if position == 1:
            model.init('Stick0.theta', output_previous['Stick.theta'])
            model.init('Stick0.phi', output_previous['Stick.phi'])
            model.init('Wstick0.w', output_previous['Wstick.w'])
            model.init('Wstick1.w', 0.0)


@cascade_builder_decorator
class BallStickStickExVivo(BallStickStick):

    name = 'BallStickStick-ExVivo (Cascade)'
    description = 'Cascade for BallStickStick with ex vivo defaults.'
    models = ('BallStick-ExVivo (Cascade)',
              'BallStickStick-ExVivo')


@cascade_builder_decorator
class BallStickStickStick(SimpleCascadeModel):

    name = 'BallStickStickStick (Cascade)'
    description = 'Cascade for BallStickStickStick.'
    models = ('BallStickStick (Cascade)',
              'BallStickStickStick')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(BallStickStickStick, self)._prepare_model(model, position, output_previous, output_all_previous)
        if position == 1:
            model.init('Wstick1.w', 0.0)


@cascade_builder_decorator
class BallStickStickStickExVivo(BallStickStickStick):

    name = 'BallStickStickStick-ExVivo (Cascade)'
    description = 'Cascade for BallStickStickStick with ex vivo defaults.'
    models = ('BallStickStick-ExVivo (Cascade)',
              'BallStickStickStick-ExVivo')