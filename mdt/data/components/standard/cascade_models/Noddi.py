from mdt.models.cascade import CascadeModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Noddi(CascadeModelBuilder):

    name = 'Noddi (Cascade)'
    description = 'Cascade for Noddi initialized from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'Noddi')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(Noddi, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'Noddi':
            model.init('Wic.w', output_previous['Wstick.w']/2.0)
            model.init('Wec.w', output_previous['Wstick.w']/2.0)
            model.init('Wcsf.w', output_previous['Wball.w'])
            model.init('Noddi_IC.theta', output_previous['Stick.theta'])
            model.init('Noddi_IC.phi', output_previous['Stick.phi'])


class NoddiS0(CascadeModelBuilder):

    name = 'Noddi (Cascade|s0)'
    description = 'Cascade for Noddi initialized with only an S0 fit.'
    models = ('s0',
              'Noddi')


class NoddiFixed(CascadeModelBuilder):

    name = 'Noddi (Cascade|fixed)'
    description = 'Cascade for Noddi with fixed directions from Ball&Stick.'
    models = ('BallStick (Cascade)',
              'Noddi')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(NoddiFixed, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'Noddi':
            model.init('Wic.w', output_previous['Wstick.w']/2.0)
            model.init('Wec.w', output_previous['Wstick.w']/2.0)
            model.init('Wcsf.w', output_previous['Wball.w'])
            model.fix('Noddi_IC.theta', output_previous['Stick.theta'])
            model.fix('Noddi_IC.phi', output_previous['Stick.phi'])


class Noddi2(CascadeModelBuilder):

    name = 'Noddi2 (Cascade)'
    description = 'Cascade for Noddi2 initialized from Ball & 2x Stick.'
    models = ('BallStickStick (Cascade)',
              'Noddi2')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(Noddi2, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'Noddi2':
            model.init('Wic0.w', output_previous['Wstick0.w']/2.0)
            model.init('Wec0.w', output_previous['Wstick0.w']/2.0)
            model.init('Wic1.w', output_previous['Wstick1.w']/2.0)
            model.init('Wec1.w', output_previous['Wstick1.w']/2.0)
            model.init('Wcsf.w', output_previous['Wball.w'])
            model.init('Noddi_IC0.theta', output_previous['Stick0.theta'])
            model.init('Noddi_IC0.phi', output_previous['Stick0.phi'])
            model.init('Noddi_IC1.theta', output_previous['Stick1.theta'])
            model.init('Noddi_IC1.phi', output_previous['Stick1.phi'])


class Noddi2Fixed(CascadeModelBuilder):

    name = 'Noddi2 (Cascade|fixed)'
    description = 'Cascade for Noddi2 with fixed directions from Ball & 2x Stick.'
    models = ('BallStickStick (Cascade)',
              'Noddi2')

    def _prepare_model(self, model, output_previous, output_all_previous):
        super(Noddi2Fixed, self)._prepare_model(model, output_previous, output_all_previous)
        if model.name == 'Noddi2':
            model.init('Wic0.w', output_previous['Wstick0.w']/2.0)
            model.init('Wec0.w', output_previous['Wstick0.w']/2.0)
            model.init('Wic1.w', output_previous['Wstick1.w']/2.0)
            model.init('Wec1.w', output_previous['Wstick1.w']/2.0)
            model.init('Wcsf.w', output_previous['Wball.w'])
            model.fix('Noddi_IC0.theta', output_previous['Stick0.theta'])
            model.fix('Noddi_IC0.phi', output_previous['Stick0.phi'])
            model.fix('Noddi_IC1.theta', output_previous['Stick1.theta'])
            model.fix('Noddi_IC1.phi', output_previous['Stick1.phi'])