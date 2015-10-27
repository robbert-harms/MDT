from mdt.components_loader import CompartmentModelsLoader
from mdt.models.single import DMRISingleModelBuilder, DMRISingleModel
from mot.evaluation_models import SumOfSquares
from mot.signal_noise_models import JohnsonSignalNoise
from mot.trees import CompartmentModelTree

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    models = []
    for x in range(1, 4):
        models.append(get_tensor_sticks(x))
    return models

compartments_loader = CompartmentModelsLoader()


def get_tensor_sticks(nmr_sticks=1):
    name = 'Tensor' + ('Stick' * nmr_sticks)
    description = 'The Ball and Stick model with {0} Sticks'.format(nmr_sticks)

    def model_construction_cb(evaluation_model=SumOfSquares(), signal_noise_model=JohnsonSignalNoise()):
        hin = (compartments_loader.get_class('Weight')('w_hin'),
               compartments_loader.load('Tensor'),
               '*')

        res = []
        for i in range(nmr_sticks):
            res.append((compartments_loader.get_class('Weight')('w_res' + repr(i)),
                        compartments_loader.get_class('Stick')('Stick' + repr(i)),
                        '*'))

        if nmr_sticks == 1:
            res = res[0]
        else:
            res.append('+')

        ml = (compartments_loader.load('S0'), (hin, res, '+'), '*')

        model = DMRISingleModel(name, CompartmentModelTree(ml),
                                         evaluation_model, signal_noise_model)

        def calculate_fr(d):
            fr = d['w_res0.w']
            for m in range(1, nmr_sticks):
                fr += d['w_res' + repr(m) + '.w']
            return fr
        modifiers = [('FR.fr', calculate_fr)]
        model.add_post_optimization_modifiers(modifiers)

        return model

    return [model_construction_cb,
            {'name': name,
             'in_vivo_suitable': True,
             'ex_vivo_suitable': False,
             'description': description}]