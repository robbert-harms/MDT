from mdt.components_loader import CompartmentModelsLoader
from mdt.dmri_composite_model import DMRICompositeSampleModel
from mot.evaluation_models import SumOfSquares
from mot.parameter_functions.transformations import SinSqrClampTransform
from mot.signal_noise_models import JohnsonSignalNoise
from mot.trees import CompartmentModelTree

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    models = []
    for x in range(1, 4):
        models.append(get_charmed(x))
    return models

compartments_loader = CompartmentModelsLoader()


def get_charmed(nmr_restr=3):
    if nmr_restr == 3:
        name = 'Charmed'
    else:
        name = 'Charmed_' + repr(nmr_restr) + 'r'

    description = 'The standard charmed model, with {0} restricted compartments.'.format(nmr_restr)

    def model_construction_cb(evaluation_model=SumOfSquares(), signal_noise_model=JohnsonSignalNoise()):
        hin = (compartments_loader.get_class('Weight')('w_hin0'),
               compartments_loader.load('Tensor').ub('d', 5e-9).lb('d', 1e-9)
                   .ub('dperp0', 5e-9).lb('dperp0', 0.3e-9)
                   .ub('dperp1', 3e-9).lb('dperp0', 0.3e-9)
                   .init('d', 1.2e-9)
                   .init('dperp0', 0.5e-9)
                   .init('dperp1', 0.5e-9),
               '*')

        hin[1].get_parameter_by_name('dperp0').parameter_transform = SinSqrClampTransform()
        hin[1].get_parameter_by_name('dperp1').parameter_transform = SinSqrClampTransform()

        if nmr_restr == 1:
            res = (compartments_loader.get_class('Weight')('w_res0'),
                   compartments_loader.get_class('CharmedRestricted')('CharmedRestricted0')
                        .lb('d', 0.3e-9).ub('d', 3e-9).init('d', 1e-9),
                   '*')
        else:
            res = []
            for i in range(nmr_restr):
                res.append((compartments_loader.get_class('Weight')('w_res' + repr(i)),
                            compartments_loader.get_class('CharmedRestricted')('CharmedRestricted' + repr(i))
                                .lb('d', 0.3e-9)
                                .ub('d', 3e-9)
                                .init('d', 1e-9 if i == i else 0.5e-9),
                            '*'))
            res.append('+')

        ml = (compartments_loader.load('S0'), (hin, res, '+'), '*')

        model = DMRICompositeSampleModel(name, CompartmentModelTree(ml),
                                         evaluation_model, signal_noise_model)
        modifiers = [('FR', lambda results: 1 - results['w_hin0.w'])]
        model.add_post_optimization_modifiers(modifiers)

        return model

    return [model_construction_cb,
            {'name': name,
             'in_vivo_suitable': True,
             'ex_vivo_suitable': False,
             'description': description}]