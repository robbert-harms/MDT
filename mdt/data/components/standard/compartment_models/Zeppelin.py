from mdt.components_config.compartment_models import CompartmentConfig
from mdt.utils import spherical_to_cartesian

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Zeppelin(CompartmentConfig):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'theta', 'phi')
    cl_code = '''
        return exp(-b *
                    (((d - dperp0) * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                                   sin(phi) * sin(theta), cos(theta), 0.0)), 2))
                     + dperp0)
                  );
    '''
    post_optimization_modifiers = [('vec0', lambda results: spherical_to_cartesian(results['theta'], results['phi']))]
