from mdt.components_config.compartment_models import CompartmentConfig
from mdt.utils import spherical_to_cartesian
from mot.model_building.cl_functions.library_functions import CerfDawson

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI_EC(CompartmentConfig):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'theta', 'phi', 'kappa')
    dependency_list = (CerfDawson(),)
    post_optimization_modifiers = [('vec0', lambda results: spherical_to_cartesian(results['theta'], results['phi']))]
