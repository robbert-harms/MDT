import numpy as np
from mdt.components_config.compartment_models import CompartmentConfig
from mdt.cl_routines.mapping.dti_measures import DTIMeasures
from mdt.utils import eigen_vectors_from_tensor

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_dti_measures_modifier():
    measures_calculator = DTIMeasures()
    return_names = measures_calculator.get_output_names()

    def modifier_routine(results_dict):
        eigen_vectors = eigen_vectors_from_tensor(results_dict['theta'], results_dict['phi'], results_dict['psi'])

        eigen_values = np.atleast_2d(np.squeeze(np.dstack([results_dict['d'],
                                                           results_dict['dperp0'],
                                                           results_dict['dperp1']])))

        measures = measures_calculator.calculate(eigen_values, eigen_vectors)
        return [measures[name] for name in return_names]

    return return_names, modifier_routine


class Tensor(CompartmentConfig):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi')
    dependency_list = ['TensorSphericalToCartesian']
    cl_code = '''
        mot_float_type4 vec0, vec1, vec2;
        TensorSphericalToCartesian(theta, phi, psi, &vec0, &vec1, &vec2);

        return exp(-b * (d *      pown(dot(vec0, g), 2) +
                         dperp0 * pown(dot(vec1, g), 2) +
                         dperp1 * pown(dot(vec2, g), 2)
                         )
                   );
    '''
    prior = 'return dperp1 < dperp0 && dperp0 < d;'
    post_optimization_modifiers = [get_dti_measures_modifier()]
