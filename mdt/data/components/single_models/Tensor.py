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
    return [get_tensor(invivo=True), get_tensor(invivo=False)]

compartments_loader = CompartmentModelsLoader()


def get_tensor(invivo=True):
    tensor_ml = (compartments_loader.load('S0'),
                 compartments_loader.load('Tensor'),
                 '*')

    name = 'Tensor'
    vivo_type = 'in-vivo'
    if not invivo:
        name += '-ExVivo'
        tensor_ml[1].init('d', 0.6e-9).init('dperp0', 0.6e-10).init('dperp1', 0.6e-11)
        vivo_type = 'ex-vivo'

    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', math.sqrt(0.5)),
                              signal_noise_model=None):
        return DMRICompositeSampleModel(name, CompartmentModelTree(tensor_ml), evaluation_model, signal_noise_model)

    return {'model_constructor': model_construction_cb,
            'name': name,
            'in_vivo_suitable': invivo,
            'ex_vivo_suitable': not invivo,
            'description': 'The standard Tensor model scaled by s0 and with {} defaults.'.format(vivo_type)}