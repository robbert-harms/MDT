import numpy as np
import itertools

from mdt.lib.post_processing import DTIMeasures, DKIMeasures, noddi_dti_maps

from mdt.model_building.parameter_functions.priors import AlwaysOne, UniformWithinBoundsPrior
from mdt.component_templates.parameters import FreeParameterTemplate, ParameterBuilder
from mdt.component_templates.compartment_models import CompartmentTemplate
from mdt.model_building.parameter_functions.transformations import IdentityTransform, PositivityTransform

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
    _lower_bound = -np.inf
    _parameter_transform = IdentityTransform()
    _sampling_prior = AlwaysOne()
    if len(set(index)) == 1:
        _lower_bound = 0
        _parameter_transform = PositivityTransform()
        _sampling_prior = UniformWithinBoundsPrior()

    class matrix_element_param(FreeParameterTemplate):
        name = 'W_{}{}{}{}'.format(*index)
        init_value = 0
        lower_bound = _lower_bound
        upper_bound = np.inf
        parameter_transform = _parameter_transform
        sampling_prior = _sampling_prior
        sampling_proposal_std = 0.01

    return ParameterBuilder().create_class(matrix_element_param)()


def get_parameters():
    """Get the list of parameters for the Kurtosis model.

    Returns:
        list: a list of parameters, some as a string some as actual parameters.
    """
    parameters = ['g', 'b', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi']

    for index in get_symmetric_indices(3, 4):
        parameters.append(build_param(index))

    return tuple(parameters)


class KurtosisTensor(CompartmentTemplate):

    parameters = get_parameters()
    dependencies = ['TensorApparentDiffusion', 'KurtosisMultiplication']
    cl_code = '''
        mot_float_type adc = TensorApparentDiffusion(theta, phi, psi, d, dperp0, dperp1, g);

        if(adc <= 0.0){
            return 1;
        }

        mot_float_type tensor_md_2 = pown((d + dperp0 + dperp1) / 3.0, 2);

        double kurtosis_sum = KurtosisMultiplication(
            W_0000, W_1111, W_2222, W_1000, W_2000, W_1110,
            W_2220, W_2111, W_2221, W_1100, W_2200, W_2211,
            W_2100, W_2110, W_2210, g);

        if(kurtosis_sum < 0 || (((tensor_md_2 * b) / adc) * kurtosis_sum) > 3.0){
            return INFINITY;
        }

        return exp(-b*adc + (b*b)/6.0 * tensor_md_2 * kurtosis_sum);
    '''

    extra_prior = 'return dperp1 < dperp0 && dperp0 < d;'

    auto_add_cartesian_vector = False
    post_optimization_modifiers = [
        DTIMeasures.post_optimization_modifier
    ]
    extra_optimization_maps = [
        DTIMeasures.extra_optimization_maps,
        DKIMeasures.extra_optimization_maps,
        noddi_dti_maps
    ]
    extra_sampling_maps = [DTIMeasures.extra_sampling_maps]
