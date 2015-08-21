import math
from mdt.components_loader import CompartmentModelsLoader
from mdt.dmri_composite_model import DMRICompositeSampleModel
from mot.evaluation_models import GaussianEvaluationModel
from mot.parameter_functions.dependencies import SimpleAssignment
from mot.trees import CompartmentModelTree

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_components_list():
    return [get_noddi()]

compartments_loader = CompartmentModelsLoader()


def get_noddi():
    noddi_ml = (compartments_loader.load('S0'),
                ((compartments_loader.get_constructor('Weight')('Wic'),
                  compartments_loader.load('Noddi_IC').fix('d', 1.7e-9).fix('R', 0.0),
                  '*'),
                 (compartments_loader.get_constructor('Weight')('Wec'),
                  compartments_loader.load('Noddi_EC').fix('d', 1.7e-9),
                  '*'),
                 (compartments_loader.get_constructor('Weight')('Wcsf'),
                  compartments_loader.load('Ball').fix('d', 3.0e-9),
                  '*'),
                 '+'),
                '*')

    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', math.sqrt(0.5)),
                              signal_noise_model=None):
        noddi_model = DMRICompositeSampleModel('Noddi', CompartmentModelTree(noddi_ml), evaluation_model,
                                               signal_noise_model)
        noddi_model.required_nmr_shells = 2

        cutoff = 1e-2
        noddi_dependencies = (
            ('Noddi_EC.dperp0', SimpleAssignment(
                'Noddi_EC.d * (((1 - Wcsf.w) < ' + repr(cutoff) + ') ? 0.0 : (Wec.w / (1 - Wcsf.w)))')),
            ('Noddi_IC.kappa', SimpleAssignment(
                '(((1 - Wcsf.w) < '+repr(cutoff)+') ? 0.0 : Noddi_IC.kappa)',
                fixed=False)),
            ('Noddi_EC.kappa', SimpleAssignment('Noddi_IC.kappa')),
            ('Noddi_EC.theta', SimpleAssignment('Noddi_IC.theta')),
            ('Noddi_EC.phi', SimpleAssignment('Noddi_IC.phi'))
        )
        noddi_model.add_parameter_dependencies(noddi_dependencies)
        modifiers = (('Noddi_IC.kappa', lambda d: d['Noddi_IC.kappa'] / 10.0),
                     ('NDI', lambda d: d['Wic.w'] / ((1 - d['Wcsf.w']) + ((1 - d['Wcsf.w']) < cutoff))))
        noddi_model.add_post_optimization_modifiers(modifiers)

        return noddi_model

    return {'model_constructor': model_construction_cb,
            'name': 'Noddi',
            'in_vivo_suitable': True,
            'ex_vivo_suitable': False,
            'description': 'The standard Noddi model'}