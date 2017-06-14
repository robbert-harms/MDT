from mdt.components_config.compartment_models import CompartmentConfig
from mdt.utils import spherical_to_cartesian

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CHARMEDRestricted(CompartmentConfig):

    parameter_list = ('g', 'b', 'G', 'Delta', 'delta', 'TE', 'd', 'theta', 'phi')
    dependency_list = ('MRIConstants',)
    post_optimization_modifiers = [('vec0', lambda results: spherical_to_cartesian(results['theta'], results['phi']))]
