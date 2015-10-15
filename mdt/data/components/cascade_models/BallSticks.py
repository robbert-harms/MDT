from mdt.cascade_model import CascadeModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallStick(CascadeModelBuilder):

    name = 'BallStick (Cascade)'
    description = 'Cascade for Ballstick'
    models = ('s0',
              'BallStick')


class BallStickExVivo(BallStick):

    name = 'BallStick-ExVivo (Cascade)'
    description = 'Cascade for Ballstick with ex vivo defaults.'
    models = ('s0',
              'BallStick-ExVivo')


class BallStickStick(CascadeModelBuilder):

    name = 'BallStickStick (Cascade)'
    description = 'Cascade for BallStickStick.'
    models = ('BallStick (Cascade)',
              'BallStickStick')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(BallStickStick, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'BallStickStick':
            model.init('Stick0.theta', output_previous['Stick.theta'])
            model.init('Stick0.phi', output_previous['Stick.phi'])
            model.init('Wstick0.w', output_previous['Wstick.w'])
            model.init('Wstick1.w', 0.0)


class BallStickStickExVivo(BallStickStick):

    name = 'BallStickStick-ExVivo (Cascade)'
    description = 'Cascade for BallStickStick with ex vivo defaults.'
    models = ('BallStick-ExVivo (Cascade)',
              'BallStickStick-ExVivo')


class BallStickStickStick(CascadeModelBuilder):

    name = 'BallStickStickStick (Cascade)'
    description = 'Cascade for BallStickStickStick.'
    models = ('BallStickStick (Cascade)',
              'BallStickStickStick')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(BallStickStickStick, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'BallStickStickStick':
            model.init('Wstick1.w', 0.0)


class BallStickStickStickExVivo(BallStickStickStick):

    name = 'BallStickStickStick-ExVivo (Cascade)'
    description = 'Cascade for BallStickStickStick with ex vivo defaults.'
    models = ('BallStickStick-ExVivo (Cascade)',
              'BallStickStickStick-ExVivo')