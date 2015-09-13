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
    return [get_s0_model(), get_s0_t2_model(), get_s0_t2_t2_model()]


compartments_loader = CompartmentModelsLoader()


def get_s0_model():
    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', ),
                              signal_noise_model=None):
        return DMRICompositeSampleModel('s0',
                                        CompartmentModelTree((compartments_loader.load('S0'),)),
                                        evaluation_model,
                                        signal_noise_model)
    return {'model_constructor': model_construction_cb,
            'name': 's0',
            'in_vivo_suitable': True,
            'ex_vivo_suitable': True,
            'description': 'Model for the unweighted signal'}


def get_s0_t2_model():
    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', math.sqrt(0.5)),
                              signal_noise_model=None):
        s0t2_model_list = (compartments_loader.load('S0'),
                           compartments_loader.load('ExpT2Dec'),
                           '*')
        return DMRICompositeSampleModel('s0-T2', CompartmentModelTree(s0t2_model_list), evaluation_model,
                                        signal_noise_model)

    return {'model_constructor': model_construction_cb,
            'name': 's0-T2',
            'in_vivo_suitable': True,
            'ex_vivo_suitable': True,
            'description': 'Model for the unweighted signal with one T2 model'}


def get_s0_t2_t2_model():
    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', math.sqrt(0.5)),
                              signal_noise_model=None):
        s0t2t2_ml = (compartments_loader.load('S0'),
                     ((compartments_loader.get_class('Weight')('Wlong'),
                       compartments_loader.get_class('ExpT2Dec')('T2long').fix('T2', 0.5),
                       '*'),
                      (compartments_loader.get_class('Weight')('Wshort'),
                       compartments_loader.get_class('ExpT2Dec')('T2short').ub('T2', 0.08),
                       '*'),
                      '+'),
                     '*')
        model = DMRICompositeSampleModel('s0-T2T2', CompartmentModelTree(s0t2t2_ml), evaluation_model,
                                         signal_noise_model)
        extra_maps = (('T2short.T2Weighted', lambda d: d['Wshort.w'] * d['T2short.T2']),
                      ('T2long.T2Weighted', lambda d: d['Wlong.w'] * d['T2long.T2']),
                      ('T2.T2', lambda d: d['T2short.T2Weighted'] + d['T2long.T2Weighted']))
        model.add_post_optimization_modifiers(extra_maps)

    return {'model_constructor': model_construction_cb,
            'name': 's0-T2T2',
            'in_vivo_suitable': True,
            'ex_vivo_suitable': True,
            'description': 'Model for the unweighted signal with two T2 models, one for short T2 and one for long T2'}
