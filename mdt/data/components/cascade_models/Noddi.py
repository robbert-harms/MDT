from mdt.cascade_model import SimpleCascadeBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [Noddi().build(),
            Noddi_fixed().build()]


class Noddi(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Noddi (Cascade)'

    def _get_description(self):
        return 'Cascade for Noddi initialized from Ball&Stick.'

    def _get_cascade_names(self):
        return ('BallStick (Cascade)',
                'Noddi')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 2:
                model.init('Wic.w', output_previous['Wstick.w']/2.0)
                model.init('Wec.w', output_previous['Wstick.w']/2.0)
                model.init('Wcsf.w', output_previous['Wball.w'])
                model.init('Noddi_IC.theta', output_previous['Stick.theta'])
                model.init('Noddi_IC.phi', output_previous['Stick.phi'])
        return _prepare_model


class Noddi_fixed(SimpleCascadeBuilder):

    def _get_name(self):
        return 'Noddi (Cascade|fixed)'

    def _get_description(self):
        return 'Cascade for Noddi with fixed directions from Ball&Stick.'

    def _get_cascade_names(self):
        return ('BallStick (Cascade)',
                'Noddi')

    def _get_prepare_model_function(self):
        def _prepare_model(self, model, position, output_previous, output_all_previous):
            if position == 2:
                model.init('Wic.w', output_previous['Wstick.w']/2.0)
                model.init('Wec.w', output_previous['Wstick.w']/2.0)
                model.init('Wcsf.w', output_previous['Wball.w'])
                model.fix('Noddi_IC.theta', output_previous['Stick.theta'])
                model.fix('Noddi_IC.phi', output_previous['Stick.phi'])
        return _prepare_model