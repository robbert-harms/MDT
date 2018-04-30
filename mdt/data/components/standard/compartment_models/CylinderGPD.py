from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CylinderGPD(CompartmentTemplate):

    parameters = ('g', 'b', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'R')
    dependencies = ('MRIConstants',
                    'VanGelderenCylinderRestricted',
                    'SphericalToCartesian')
    cl_code = '''
        const mot_float_type direction_2 = pown(dot(g, SphericalToCartesian(theta, phi)), 2);

        const mot_float_type signal_par = -b * d * direction_2;
        const mot_float_type signal_perp = (1 - direction_2) * VanGelderenCylinderRestricted(G, Delta, delta, d, R);
        
        return exp(signal_perp + signal_par);
    '''
