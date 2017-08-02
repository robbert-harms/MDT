import numpy as np
import itertools

from mdt.cl_routines.mapping.dki_measures import DKIMeasures
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mdt.component_templates.parameters import FreeParameterTemplate, ParameterBuilder
from mdt.component_templates.compartment_models import CompartmentTemplate
from mot.model_building.parameter_functions.transformations import IdentityTransform

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
    """Get one of the parameters of the Kurtosis matrix as an object.

    Args:
        tuple: index, 4 integers specyfing the location of this parameter in the matrix.

    Returns:
        Parameter: a constructed parameter to be used in a compartment model.
    """
    class matrix_element_param(FreeParameterTemplate):
        name = 'W_{}{}{}{}'.format(*index)
        init_value = 0
        lower_bound = -np.inf
        upper_bound = np.inf
        parameter_transform = IdentityTransform()
        sampling_proposal = GaussianProposal(0.01)
    return ParameterBuilder().create_class(matrix_element_param)()


def get_parameter_list():
    """Get the list of parameters for the Kurtosis model.

    Returns:
        list: a list of parameters, some as a string some as actual parameters.
    """
    parameter_list = ['g', 'b', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi']

    for index in get_symmetric_indices(3, 4):
        parameter_list.append(build_param(index))

    return tuple(parameter_list)


def get_dki_measures_modifier():
    """Get the DKI post processing modification routine(s)."""
    dki_calc = DKIMeasures()

    def _calculate_dki_results(results_dict):
        measures = dki_calc.calculate(results_dict)
        return [measures[name] for name in dki_calc.get_output_names()]

    def modifier_routine(results_dict):
        dki_results = _calculate_dki_results(results_dict)
        return dki_results

    return_names = dki_calc.get_output_names()
    return return_names, modifier_routine


class KurtosisExtension(CompartmentTemplate):

    description = '''
        The Kurtosis extension for the Tensor model. 
        
        This compartment can not be used directly as a compartment model, it always needs to be used in conjunction with 
        the Tensor model. For example, a composite model script would be: "S0 * Tensor * Kurtosis".
    '''
    parameter_list = get_parameter_list()
    dependency_list = ['TensorSphericalToCartesian', 'KurtosisMultiplication']
    cl_code = '''
        mot_float_type4 vec0, vec1, vec2;
        TensorSphericalToCartesian(theta, phi, psi, &vec0, &vec1, &vec2);

        mot_float_type d_app = d *      pown(dot(vec0, g), 2) +
                               dperp0 * pown(dot(vec1, g), 2) +
                               dperp1 * pown(dot(vec2, g), 2);
        
        if(d_app <= 0.0){
            return 1;
        }
        
        mot_float_type tensor_md_2 = pown((d + dperp0 + dperp1) / 3.0, 2);
        
        double kurtosis_sum = KurtosisMultiplication(
            W_0000, W_1111, W_2222, W_1000, W_2000, W_1110, 
            W_2220, W_2111, W_2221, W_1100, W_2200, W_2211, 
            W_2100, W_2110, W_2210, g);
        
        if(kurtosis_sum < 0 || (((tensor_md_2 * b) / d_app) * kurtosis_sum) > 3.0){
            return INFINITY;
        }
             
        return exp((b*b)/6.0 * tensor_md_2 * kurtosis_sum);
    '''
    post_optimization_modifiers = [get_dki_measures_modifier()]
