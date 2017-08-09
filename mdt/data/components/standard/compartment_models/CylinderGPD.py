from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CylinderGPD(CompartmentTemplate):

    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'R')
    dependency_list = ('MRIConstants',
                       'NeumannCylindricalRestrictedSignal',
                       'SphericalToCartesian')
    cl_code = '''
        mot_float_type b = pown(GAMMA_H * delta * G, 2) * (Delta - (delta/3.0));

        mot_float_type lperp = NeumannCylindricalRestrictedSignal(Delta, delta, d, R, G);

        mot_float_type gn2 = pown(dot(g, SphericalToCartesian(theta, phi)), 2);

        return exp( ((1 - gn2) * lperp) + (-b * d * gn2));
    '''
