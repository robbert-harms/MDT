import numpy as np
from mdt.component_templates.compartment_models import CompartmentTemplate
from mdt.cl_routines.mapping.dti_measures import DTIMeasures

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_dti_measures_modifier():
    measures_calculator = DTIMeasures()
    return_names = measures_calculator.get_output_names()

    def modifier_routine(results_dict):
        matrix = np.array(
            [[results_dict['D_{}{}'.format(i, j) if 'D_{}{}'.format(i, j) in results_dict else 'D_{}{}'.format(j, i)]
              for j in range(3)] for i in range(3)]).transpose()

        eigen_values, eigen_vectors = np.linalg.eig(matrix)
        measures = measures_calculator.calculate(eigen_values, eigen_vectors)
        return [measures[name] for name in return_names]

    return return_names, modifier_routine


class SymmetricNonParametricTensor(CompartmentTemplate):

    description = '''
        The Tensor model in which a symmetric D matrix is optimized directly,
        without vector/diffusivity parameterization.
    '''
    parameter_list = ('g', 'b',
                      'Tensor_D_00(D_00)', 'Tensor_D_01(D_01)', 'Tensor_D_02(D_02)',
                                           'Tensor_D_11(D_11)', 'Tensor_D_12(D_12)',
                                                                'Tensor_D_22(D_22)')
    cl_code = '''
        double diff = g.x * g.x * D_00 +
                        g.y * g.x * D_01 * 2 +
                        g.y * g.y * D_11 +
                        g.z * g.x * D_02 * 2 +
                        g.z * g.y * D_12 * 2 +
                        g.z * g.z * D_22;
        return exp(-b * diff);
    '''
    post_optimization_modifiers = [get_dti_measures_modifier()]
