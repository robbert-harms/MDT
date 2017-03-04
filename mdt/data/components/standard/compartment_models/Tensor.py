import numpy as np
from mdt.components_loader import bind_function
from mdt.components_config.compartment_models import CompartmentConfig
from mdt.cl_routines.mapping.dti_measures import DTIMeasures
from mdt.utils import eigen_vectors_from_tensor

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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

    @bind_function
    def get_extra_results_maps(self, results_dict):
        eigen_vectors = eigen_vectors_from_tensor(results_dict[self.name + '.theta'], results_dict[self.name + '.phi'],
                                                  results_dict[self.name + '.psi'])

        eigen_values = np.atleast_2d(np.squeeze(np.dstack([results_dict[self.name + '.d'],
                                                           results_dict[self.name + '.dperp0'],
                                                           results_dict[self.name + '.dperp1']])))

        measures = DTIMeasures().calculate(eigen_values, eigen_vectors)
        return {'{}.{}'.format(self.name, key): value for key, value in measures.items()}
