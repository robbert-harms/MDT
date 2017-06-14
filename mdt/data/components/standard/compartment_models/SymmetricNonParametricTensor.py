import numpy as np
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mdt.components_config.parameters import FreeParameterConfig, ParameterBuilder
from mdt.components_config.compartment_models import CompartmentConfig
from mdt.cl_routines.mapping.dti_measures import DTIMeasures
from mot.model_building.parameter_functions.transformations import SinSqrClampTransform

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_param(param_name):
    initial_values = {'D00': 0.3e-9, 'D11': 0.3e-9, 'D22': 1.2e-9, 'D01': 0, 'D02': 0, 'D12': 0}
    lower_bounds = {'D00': 0, 'D11': 0, 'D22': 0, 'D01': -1e-9, 'D02': -1e-9, 'D12': -1e-9}
    upper_bounds = {'D00': 5e-9, 'D11': 5e-9, 'D22': 5e-9, 'D01': 1e-9, 'D02':  1e-9, 'D12': 1e-9}

    class matrix_element_param(FreeParameterConfig):
        name = param_name
        init_value = initial_values[param_name] if param_name in initial_values else 0
        lower_bound = lower_bounds[param_name] if param_name in lower_bounds else 0
        upper_bound = upper_bounds[param_name] if param_name in upper_bounds else 0
        parameter_transform = SinSqrClampTransform()
        sampling_proposal = GaussianProposal(1e-10)

    return ParameterBuilder().create_class(matrix_element_param)()


def get_dti_measures_modifier():
    measures_calculator = DTIMeasures()
    return_names = measures_calculator.get_output_names()

    def modifier_routine(results_dict):
        matrix = np.array(
            [[results_dict['D{}{}'.format(i, j) if 'D{}{}'.format(i, j) in results_dict else 'D{}{}'.format(j, i)]
              for j in range(3)] for i in range(3)]).transpose()

        eigen_values, eigen_vectors = np.linalg.eig(matrix)
        measures = measures_calculator.calculate(eigen_values, eigen_vectors)
        return [measures[name] for name in return_names]

    return return_names, modifier_routine


class SymmetricNonParametricTensor(CompartmentConfig):

    description = "The Tensor model in which a symmetric D matrix is optimized directly " \
                  "(without vector/diffusivity parameterization)."

    parameter_list = ('g', 'b',
                      get_param('D00'), get_param('D01'), get_param('D02'),
                                        get_param('D11'), get_param('D12'),
                                                          get_param('D22'))
    cl_code = '''
        double diff = g.x * (g.x * D00 + g.y * D01 + g.z * D02) +
                      g.y * (g.x * D01 + g.y * D11 + g.z * D12) +
                      g.z * (g.x * D02 + g.y * D12 + g.z * D22);

        return exp(-b * diff);
    '''
    post_optimization_modifiers = [get_dti_measures_modifier()]
