import numpy as np
import itertools

from mdt.utils import eigen_vectors_from_tensor
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mdt.component_templates.parameters import FreeParameterTemplate, ParameterBuilder
from mdt.component_templates.compartment_models import CompartmentTemplate
from mdt.cl_routines.mapping.dti_measures import DTIMeasures
from mot.model_building.parameter_functions.transformations import ClampTransform

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_symmetric_indices(length, dimensions):
    """Get the indices we will use for indexing a symmetric matrix.

    This returns the indices of a lower triangular matrix.

    Args:
        length (int): the length of each side of the cube
        dimensions (int): the number of dimensions

    Returns:
        list of tuple: for every index numerical positions
    """
    def descending_or_equal(indices):
        """Predicate to decide if a index is part of the lower triangle of the matrix"""
        if len(indices) <= 1:
            return True
        if len(indices) == 2:
            return indices[0] >= indices[1]
        else:
            return (indices[0] >= indices[1]) and descending_or_equal(indices[1:])

    all_indices = list(itertools.product(*([range(length)] * dimensions)))
    return list(filter(descending_or_equal, all_indices))


def build_param(index):
    class matrix_element_param(FreeParameterTemplate):
        name = 'W_{i}{j}{k}{l}'.format(i=index[0], j=index[1], k=index[2], l=index[3])
        init_value = 0
        lower_bound = -1
        upper_bound = 1
        parameter_transform = ClampTransform()
        sampling_proposal = GaussianProposal(1e-10)

    return ParameterBuilder().create_class(matrix_element_param)()


def get_parameter_list():
    parameter_list = ['g', 'b', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi']

    for index in get_symmetric_indices(3, 4):
        parameter_list.append(build_param(index))

    return tuple(parameter_list)


def get_dki_measures_modifier():
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


class Kurtosis(CompartmentTemplate):

    description = "The Kurtosis model"
    parameter_list = get_parameter_list()
    dependency_list = ['TensorSphericalToCartesian']
    cl_code = '''
        mot_float_type4 vec0, vec1, vec2;
        TensorSphericalToCartesian(theta, phi, psi, &vec0, &vec1, &vec2);

        mot_float_type d_app = d *      pown(dot(vec0, g), 2) +
                               dperp0 * pown(dot(vec1, g), 2) +
                               dperp1 * pown(dot(vec2, g), 2);
        
        if(d_app <= 0.0){
            return 1;
        }
        
        if(b < 25e6){
            return exp(-b * d_app);
        }
        
        mot_float_type tensor_md_2 = pown((d + dperp0 + dperp1) / 3.0, 2);
        
        double kurtosis_sum = 0;
        kurtosis_sum += g.x * g.x * g.x * g.x * W_0000;
        kurtosis_sum += g.y * g.y * g.y * g.y * W_1111;
        kurtosis_sum += g.z * g.z * g.z * g.z * W_2222;
        
        kurtosis_sum += g.y * g.x * g.x * g.x * W_1000 * 4;
        kurtosis_sum += g.z * g.x * g.x * g.x * W_2000 * 4;
        kurtosis_sum += g.y * g.y * g.y * g.x * W_1110 * 4;
        kurtosis_sum += g.z * g.z * g.z * g.x * W_2220 * 4;
        kurtosis_sum += g.z * g.y * g.y * g.y * W_2111 * 4;
        kurtosis_sum += g.z * g.z * g.z * g.y * W_2221 * 4;
        
        kurtosis_sum += g.y * g.y * g.x * g.x * W_1100 * 6;
        kurtosis_sum += g.z * g.z * g.x * g.x * W_2200 * 6;
        kurtosis_sum += g.z * g.z * g.y * g.y * W_2211 * 6;
        
        kurtosis_sum += g.z * g.y * g.x * g.x * W_2100 * 12;
        kurtosis_sum += g.z * g.y * g.y * g.x * W_2110 * 12;
        kurtosis_sum += g.z * g.z * g.y * g.x * W_2210 * 12;
        
        if(kurtosis_sum <= -2 || (((tensor_md_2 * b) / d_app) * kurtosis_sum) > 3.0){
            return INFINITY;
        }
             
        return exp(-b * d_app + (b*b)/6.0 * tensor_md_2 * kurtosis_sum);
    '''
    post_optimization_modifiers = [get_dki_measures_modifier()]
