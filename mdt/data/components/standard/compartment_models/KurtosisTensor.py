import numpy as np
import itertools

from mdt.cl_routines.mapping.dki_measures import DKIMeasures
from mdt.cl_routines.mapping.dti_measures import DTIMeasures
from mdt.utils import tensor_spherical_to_cartesian

from mot.model_building.parameter_functions.priors import AlwaysOne, UniformWithinBoundsPrior
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mdt.component_templates.parameters import FreeParameterTemplate, ParameterBuilder
from mdt.component_templates.compartment_models import CompartmentTemplate
from mot.model_building.parameter_functions.transformations import IdentityTransform, PositiveTransform

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
        _parameter_transform = PositiveTransform()
        _sampling_prior = UniformWithinBoundsPrior()

    class matrix_element_param(FreeParameterTemplate):
        name = 'W_{}{}{}{}'.format(*index)
        init_value = 0
        lower_bound = _lower_bound
        upper_bound = np.inf
        parameter_transform = _parameter_transform
        sampling_prior = _sampling_prior
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

    def modifier_routine(results_dict):
        measures = dki_calc.calculate(results_dict)
        return [measures[name] for name in dki_calc.get_output_names()]

    return_names = dki_calc.get_output_names()
    return return_names, modifier_routine


def get_dti_measures_modifier():
    measures_calculator = DTIMeasures()
    return_names = measures_calculator.get_output_names()

    def modifier_routine(results_dict):
        measures = measures_calculator.calculate(results_dict)
        return [measures[name] for name in return_names]

    return return_names, modifier_routine


def extra_covariance_samples(theta, phi, psi):
    return np.rollaxis(np.concatenate(tensor_spherical_to_cartesian(theta, phi, psi), axis=2), 2, 1)


class KurtosisTensor(CompartmentTemplate):

    description = '''
        The Kurtosis Tensor model.
    '''
    parameter_list = get_parameter_list()
    dependency_list = ['TensorApparentDiffusion', 'KurtosisMultiplication']
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
    post_optimization_modifiers = [get_dti_measures_modifier(),
                                   get_dki_measures_modifier()]

    auto_sampling_covar_cartesian = False
    sampling_covar_extras = [(('theta', 'phi', 'psi'), ('vec0_x', 'vec0_y', 'vec0_z',
                                                        'vec1_x', 'vec1_y', 'vec1_z',
                                                        'vec2_x', 'vec2_y', 'vec2_z'), extra_covariance_samples)]
    sampling_covar_exclude = ['theta', 'phi', 'psi']
