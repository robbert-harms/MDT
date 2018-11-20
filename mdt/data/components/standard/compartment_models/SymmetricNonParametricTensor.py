import numpy as np
from mdt import CompartmentTemplate, FreeParameterTemplate
from mdt.lib.post_processing import DTIMeasures
from mdt.model_building.parameter_functions.transformations import ScaleTransform

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def extra_dti_results(results_dict):
    output = {}

    matrix = np.array(
        [[results_dict['D_{}{}'.format(i, j) if 'D_{}{}'.format(i, j) in results_dict else 'D_{}{}'.format(j, i)]
          for j in range(3)] for i in range(3)]).transpose()

    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    eigen_values = np.maximum(eigen_values, 0)

    output.update({'vec{}'.format(ind): eigen_vectors[:, ind] for ind in range(3)})
    output['d'] = eigen_values[:, 0]
    output['dperp0'] = eigen_values[:, 1]
    output['dperp1'] = eigen_values[:, 2]

    output.update(DTIMeasures.extra_optimization_maps(output))

    return output


class SymmetricNonParametricTensor(CompartmentTemplate):
    """The Tensor model in which a symmetric D matrix is optimized directly.

    This does not use the vector/diffusivity parameterization using the eigenvectors/eigenvalues.
    """
    parameters = ('g', 'b',
                  'D_00', 'D_01', 'D_02',
                          'D_11', 'D_12',
                                  'D_22')
    cl_code = '''
        double diff = g.x * g.x * D_00 +
                      g.y * g.x * D_01 * 2 +
                      g.y * g.y * D_11 +
                      g.z * g.x * D_02 * 2 +
                      g.z * g.y * D_12 * 2 +
                      g.z * g.z * D_22;
        return exp(-b * diff);
    '''
    extra_optimization_maps = [extra_dti_results]

    class D_00(FreeParameterTemplate):
        init_value = 0.3e-9
        lower_bound = 0
        upper_bound = 5e-9
        parameter_transform = ScaleTransform(1e10)
        sampling_proposal_std = 1e-10
        numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}

    class D_11(FreeParameterTemplate):
        init_value = 0.3e-9
        lower_bound = 0
        upper_bound = 5e-9
        parameter_transform = ScaleTransform(1e10)
        sampling_proposal_std = 1e-10
        numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}

    class D_22(FreeParameterTemplate):
        init_value = 1.2e-9
        lower_bound = 0
        upper_bound = 5e-9
        parameter_transform = ScaleTransform(1e10)
        sampling_proposal_std = 1e-10
        numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}

    class D_01(FreeParameterTemplate):
        init_value = 0.1e-9
        lower_bound = 0
        upper_bound = 5e-9
        parameter_transform = ScaleTransform(1e10)
        sampling_proposal_std = 1e-10
        numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}

    class D_02(FreeParameterTemplate):
        init_value = 0.1e-9
        lower_bound = 0
        upper_bound = 5e-9
        parameter_transform = ScaleTransform(1e10)
        sampling_proposal_std = 1e-10
        numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}

    class D_12(FreeParameterTemplate):
        init_value = 0.1e-9
        lower_bound = 0
        upper_bound = 5e-9
        parameter_transform = ScaleTransform(1e10)
        sampling_proposal_std = 1e-10
        numdiff_info = {'scale_factor': 1e10, 'use_upper_bound': False}
