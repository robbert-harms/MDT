import math
from mdt.components_loader import CompartmentModelsLoader
from mdt.dmri_composite_model import DMRICompositeSampleModel
from mot.evaluation_models import GaussianEvaluationModel
from mot.trees import CompartmentModelTree

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [get_tensor_t2(True), get_tensor_t2(False)]

compartments_loader = CompartmentModelsLoader()


def get_tensor_t2(invivo=True):
    name = 'Tensor-T2'
    vivo_type = 'in-vivo'
    if invivo:
        d = 1.7e-9
        dperp0 = 1.7e-10
        dperp1 = 1.7e-11
    else:
        name += '-ExVivo'
        vivo_type = 'ex-vivo'
        d = 0.6e-9
        dperp0 = 0.6e-10
        dperp1 = 0.6e-11

    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', 1),
                              signal_noise_model=None):
        tensor_ml = (compartments_loader.load('S0'),
                     compartments_loader.load('ExpT2Dec'),
                     compartments_loader.load('Tensor').init('d', d)
                                                       .init('dperp0', dperp0)
                                                       .init('dperp1', dperp1),
                     '*')

        return DMRICompositeSampleModel(name, CompartmentModelTree(tensor_ml), evaluation_model, signal_noise_model)

    return {'model_constructor': model_construction_cb,
            'name': name,
            'in_vivo_suitable': invivo,
            'ex_vivo_suitable': not invivo,
            'description': 'The Tensor model scaled by s0 and T2 and with {} defaults.'.format(vivo_type)}