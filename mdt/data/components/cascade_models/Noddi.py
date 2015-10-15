from mdt.cascade_model import cascade_builder_decorator, SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [Noddi.get_meta_data(),
            NoddiFixed.get_meta_data()]


@cascade_builder_decorator
class Noddi(SimpleCascadeModel):

    name = 'Noddi (Cascade)'
    description = 'Cascade for Noddi initialized from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'Noddi')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(Noddi, self)._prepare_model(model, position, output_previous, output_all_previous)
        if position == 1:
            model.init('Wic.w', output_previous['Wstick.w']/2.0)
            model.init('Wec.w', output_previous['Wstick.w']/2.0)
            model.init('Wcsf.w', output_previous['Wball.w'])
            model.init('Noddi_IC.theta', output_previous['Stick.theta'])
            model.init('Noddi_IC.phi', output_previous['Stick.phi'])


@cascade_builder_decorator
class NoddiFixed(SimpleCascadeModel):

    name = 'Noddi (Cascade|fixed)'
    description = 'Cascade for Noddi with fixed directions from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'Noddi')

    def _prepare_model(self, model, position, output_previous, output_all_previous):
        super(NoddiFixed, self)._prepare_model(model, position, output_previous, output_all_previous)
        if position == 1:
            model.init('Wic.w', output_previous['Wstick.w']/2.0)
            model.init('Wec.w', output_previous['Wstick.w']/2.0)
            model.init('Wcsf.w', output_previous['Wball.w'])
            model.fix('Noddi_IC.theta', output_previous['Stick.theta'])
            model.fix('Noddi_IC.phi', output_previous['Stick.phi'])