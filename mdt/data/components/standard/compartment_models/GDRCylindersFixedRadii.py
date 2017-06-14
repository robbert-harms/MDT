from mdt.components_config.compartment_models import CompartmentConfig
from mdt.components_loader import CompartmentModelsLoader
from mdt.utils import spherical_to_cartesian

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


compartment_loader = CompartmentModelsLoader()


class GDRCylindersFixedRadii(CompartmentConfig):

    description = '''
        Generate the compartment model signal for the Gamma Distributed Radii model.

        This is a fixed version of the GDRCylinders model. This means that the different radii are not calculated
        dynamically by means of a Gamma distribution. Rather, the list of radii and the corresponding weights
        are given as fixed values.

        Args:
            gamma_cyl_radii: the list of radii that should be used for calculating the cylinders.
            gamma_cyl_weights: the list of weights per radius.
            nmr_gamma_cyl: the number of cylinders we provided
    '''
    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'gamma_radii',
                      'gamma_cyl_weights', 'nmr_gamma_cyl_weights')
    dependency_list = (compartment_loader.load('CylinderGPD'),)
    cl_code = '''
        double signal = 0;
        for(int i = 0; i < nmr_gamma_cyl_fixed; i++){
            signal += gamma_cyl_weights[i] * cmCylinderGPD(g, G, Delta, delta, d, theta, phi, gamma_cyl_radii[i]);
        }
        return signal;
    '''
    post_optimization_modifiers = [('vec0', lambda results: spherical_to_cartesian(results['theta'], results['phi']))]
