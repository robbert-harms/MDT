from mdt.components_loader import CompartmentModelsLoader
from mdt.dmri_composite_model import DMRICompositeSampleModel
from mot.evaluation_models import SumOfSquares
from mot.parameter_functions.dependencies import SimpleAssignment
from mot.trees import CompartmentModelTree

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [get_activeax()]

compartments_loader = CompartmentModelsLoader()


def get_activeax():
    name = 'ActiveAx'

    def model_construction_cb(evaluation_model=SumOfSquares(), signal_noise_model=None):
        active_ax_ml = (compartments_loader.load('S0'),
                        ((compartments_loader.get_class('Weight')('wic'),
                          compartments_loader.load('CylinderGPD').fix('d', 1.7e-9),
                          '*'),
                         (compartments_loader.get_class('Weight')('wec'),
                          compartments_loader.load('Zeppelin').fix('d', 1.7e-9),
                          '*'),
                         (compartments_loader.get_class('Weight')('wcsf'),
                          compartments_loader.load('Ball').fix('d', 3e-9),
                          '*'),
                         '+'),
                        '*')

        model = DMRICompositeSampleModel(name, CompartmentModelTree(active_ax_ml), evaluation_model, signal_noise_model)
        model.add_parameter_dependency('Zeppelin.dperp0', SimpleAssignment('Zeppelin.d * (wec.w / (wec.w + wic.w))'))
        return model

    return {'model_constructor': model_construction_cb,
            'name': name,
            'in_vivo_suitable': True,
            'ex_vivo_suitable': True,
            'description': 'The standard ActiveAx model.'}