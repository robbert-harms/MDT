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
    def model_construction_cb(evaluation_model=GaussianEvaluationModel().fix('sigma', 1),
                              signal_noise_model=None):
        noddi_ml = (compartments_loader.load('S0'),
                    ((compartments_loader.get_class('Weight')('Wic'),
                      compartments_loader.load('Noddi_IC').fix('d', 1.7e-9).fix('R', 0.0),
                      '*'),
                     (compartments_loader.get_class('Weight')('Wec'),
                      compartments_loader.load('Noddi_EC').fix('d', 1.7e-9),
                      '*'),
                     (compartments_loader.get_class('Weight')('Wcsf'),
                      compartments_loader.load('Ball').fix('d', 3.0e-9),
                      '*'),
                     '+'),
                    '*')

        noddi_model = DMRICompositeSampleModel('Noddi', CompartmentModelTree(noddi_ml), evaluation_model,
                                               signal_noise_model)

        eps = 1e-5
        cutoff = 0.01
        noddi_dependencies = (
            ('Noddi_EC.dperp0', SimpleAssignment('Noddi_EC.d * (Wec.w / (1 - Wcsf.w + {}))'.format(eps))),
            ('Noddi_IC.kappa', SimpleAssignment('((1 - Wcsf.w) >= {}) * Noddi_IC.kappa'.format(cutoff),
                                                fixed=False)),
            ('Noddi_EC.kappa', SimpleAssignment('Noddi_IC.kappa')),
            ('Noddi_EC.theta', SimpleAssignment('Noddi_IC.theta')),
            ('Noddi_EC.phi', SimpleAssignment('Noddi_IC.phi'))
        )
        noddi_model.add_parameter_dependencies(noddi_dependencies)
        modifiers = (('NDI', lambda d: d['Wic.w'] / (d['Wic.w'] + d['Wec.w'])),
                     ('SNIF', lambda d: 1 - d['Wcsf.w']),
                     ('ODI', lambda d: d['Noddi_IC.odi']))
        noddi_model.add_post_optimization_modifiers(modifiers)

        return noddi_model

    return {'model_constructor': model_construction_cb,
            'name': 'Noddi',
            'in_vivo_suitable': True,
            'ex_vivo_suitable': False,
            'description': 'The standard Noddi model'}