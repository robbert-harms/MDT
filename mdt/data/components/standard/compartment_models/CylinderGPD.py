from mdt.components_config.compartment_models import CompartmentConfig
from mdt.utils import spherical_to_cartesian

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CylinderGPD(CompartmentConfig):

    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'R')
    dependency_list = ('MRIConstants',
                       'NeumannCylindricalRestrictedSignal')
    cl_code = '''
        mot_float_type b = pown(GAMMA_H * delta * G, 2) * (Delta - (delta/3.0));
        
        mot_float_type lperp = NeumannCylindricalRestrictedSignal(Delta, delta, d, R, G);
        
        mot_float_type gn2 = pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta), 
                                                           sin(phi) * sin(theta), cos(theta), 0.0)), 2);
        
        return exp( ((1 - gn2) * lperp) + (-b * d * gn2));
    '''
    post_optimization_modifiers = [('vec0', lambda results: spherical_to_cartesian(results['theta'], results['phi']))]
