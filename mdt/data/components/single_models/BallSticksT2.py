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
    models = []
    for x in range(1, 4):
        models.append(get_ball_sticks_t2(x, invivo=True))
        models.append(get_ball_sticks_t2(x, invivo=False))
        models.append(get_ball_sticks_t2t2(x, invivo=True))
        models.append(get_ball_sticks_t2t2(x, invivo=False))
    return models


def get_ball_sticks_t2(nmr_sticks=1, invivo=True):
    name = 'Ball' + ('Stick' * nmr_sticks) + '-T2'
    if invivo:
        d_ani = 1.7e-9
        d_iso = 3.0e-9
        vivo_type = 'in-vivo'
    else:
        d_ani = 0.6e-9
        d_iso = 2.0e-9
        name += '-ExVivo'
        vivo_type = 'ex-vivo'

    description = 'The Ball and Stick model with {0} Sticks, single T2 weighted and with {1} defaults.'.\
        format(nmr_sticks, vivo_type)

    compartments_loader = CompartmentModelsLoader()

    csf = (compartments_loader.get_constructor('Weight')('Wball'),
           compartments_loader.load('Ball').fix('d', d_iso),
           '*')

    if nmr_sticks == 1:
        ic = (compartments_loader.get_constructor('Weight')('Wstick'),
              compartments_loader.load('Stick').fix('d', d_ani),
              '*')
    else:
        ic = []
        for i in range(nmr_sticks):
            ic.append((compartments_loader.get_constructor('Weight')('Wstick' + repr(i)),
                       compartments_loader.get_constructor('Stick')('Stick' + repr(i)).fix('d', d_ani),
                       '*'))
        ic.append('+')

    ml = (compartments_loader.load('S0'),
          compartments_loader.load('ExpT2Dec'),
          (csf, ic, '+'), '*')

    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', math.sqrt(0.5)),
                              signal_noise_model=None):
        return DMRICompositeSampleModel(name, CompartmentModelTree(ml), evaluation_model, signal_noise_model)

    return {'model_constructor': model_construction_cb,
            'name': name,
            'in_vivo_suitable': invivo,
            'ex_vivo_suitable': not invivo,
            'description': description}


def get_ball_sticks_t2t2(nmr_sticks=1, invivo=True):
    name = 'Ball' + ('Stick' * nmr_sticks) + '-T2T2'
    if invivo:
        d_ani = 1.7e-9
        d_iso = 3.0e-9
        vivo_type = 'in-vivo'
    else:
        d_ani = 0.6e-9
        d_iso = 2.0e-9
        name += '-ExVivo'
        vivo_type = 'ex-vivo'

    description = 'The Ball and Stick model with {0} Sticks, one short T2 and one long T2, using {1} defaults.'.\
        format(nmr_sticks, vivo_type)

    compartments_loader = CompartmentModelsLoader()

    csf = (compartments_loader.get_constructor('ExpT2Dec')('T2long'),
           compartments_loader.get_constructor('Weight')('Wball'),
           compartments_loader.load('Ball').fix('d', d_iso),
           '*')

    if nmr_sticks == 1:
        ic = (compartments_loader.get_constructor('Weight')('Wstick'),
              compartments_loader.load('Stick').fix('d', d_ani),
              '*')
    else:
        ic = []
        for i in range(nmr_sticks):
            ic.append((compartments_loader.get_constructor('Weight')('Wstick' + repr(i)),
                       compartments_loader.get_constructor('Stick')('Stick' + repr(i)).fix('d', d_ani),
                       '*'))
        ic.append('+')

    ic = (compartments_loader.get_constructor('ExpT2Dec')('T2short'), ic, '*')

    ml = (compartments_loader.load('S0'), (csf, ic, '+'), '*')

    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', math.sqrt(0.5)),
                              signal_noise_model=None):
        return DMRICompositeSampleModel(name, CompartmentModelTree(ml), evaluation_model, signal_noise_model)

    return {'model_constructor': model_construction_cb,
            'name': name,
            'in_vivo_suitable': invivo,
            'ex_vivo_suitable': not invivo,
            'description': description}