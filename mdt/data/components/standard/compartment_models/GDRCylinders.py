from mdt.components_config.compartment_models import CompartmentConfig
from mdt.components_loader import CompartmentModelsLoader
from mdt.utils import spherical_to_cartesian

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GDRCylinders(CompartmentConfig):

    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'gamma_k', 'gamma_beta', 'gamma_nmr_cyl')
    dependency_list = (CompartmentModelsLoader().load('CylinderGPD'),)
    post_optimization_modifiers = [('vec0', lambda results: spherical_to_cartesian(results['theta'], results['phi']))]
