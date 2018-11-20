"""This module contains various standard post-processing routines for use after optimization or sample."""
import numpy as np

from mdt.lib.sorting import create_2d_sort_matrix
from mdt.utils import tensor_spherical_to_cartesian, tensor_cartesian_to_spherical, \
    voxelwise_vector_matrix_vector_product, create_covariance_matrix
from mot import minimize
from mot.lib.cl_function import SimpleCLFunction
from mot.lib.utils import split_in_batches, parse_cl_function
from mot.lib.kernel_data import Array, Zeros, Scalar, Struct
from mdt.lib.components import get_component
from mot.library_functions import dawson

__author__ = 'Robbert Harms'
__date__ = '2017-12-10'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def get_sort_modifier(sort_specification):
    """Create a callback routine to sort the output maps according to the given specification.

    This is typically used to ensure that the point estimate from optimization can directly be used in MCMC sample.
    For example, some models have a prior on the weights to restrict them to a decreasing order, which is typically
    done to prevent bimodal distributions. To ensure that the optimization output is within such a prior, you will
    need to sort the weights as a post processing after optimization.

    The syntax of the sort specification is as follows::

        sort_maps = {
            <param_0>: (<map_0>, <other_map_0>, ...),
            <param_1>: (<map_1>, <other_map_1>, ...)
            ...
        }

    The keys of the dictionary provide the parameters to use for the sorting. These are typically one of the
    parameters of a similar model, like a Weight compartment. The values of the dictionary specify which
    compartments or parameters to sort based on the sorting of the keys. The length of the value tuples should
    match for each key since each index is sorted against the same index in other tuples. That is, in the specification
    above, <map_0> is sorted against <map_1>.

    As an concrete example, suppose we have the model::

        model_expression = '''
            S0 * (Weight(w_zeppelin_1) * ExpT2Dec(ExpT2Dec_zeppelin_1) * Zeppelin(Zeppelin_1) +
                  Weight(w_zeppelin_2) * ExpT2Dec(ExpT2Dec_zeppelin_2) * Zeppelin(Zeppelin_2) )
        '''

    And we wish to sort the weights and the Zeppelins based on the T2 values. To do so, we would specify::

        sort_specification = {
            'ExpT2Dec_zeppelin_1.T2': ('w_zeppelin_1', 'Zeppelin_1'),
            'ExpT2Dec_zeppelin_2.T2': ('w_zeppelin_2', 'Zeppelin_2')
        }

    This specification first creates a sort matrix based on the T2's and then sort the T2's accordingly.
    Afterwards it will sort the parameters of the Zeppelin and Weight compartments based on that sorting matrix.

    One can also sort just one parameter of a compartment using::

        sort_specification = {
            'ExpT2Dec_zeppelin_1.T2': ('w_zeppelin_1', 'Zeppelin_1.theta'),
            'ExpT2Dec_zeppelin_2.T2': ('w_zeppelin_2', 'Zeppelin_2.theta')
        }

    this will only apply the sorting on the theta map of both the Zeppelin compartments ((note the
    additional ``.theta``).

    Args:
        sort_specification: the sorting specification used in the sorting

    Returns:
        function: the post optimization modification call back function.
    """

    def is_param_name(name):
        return '.' in name

    def resolve_sort_pairs(results):
        sort_pairs = []
        for specified_pair in list(zip(*sort_specification.values())):
            if is_param_name(specified_pair[0]):
                sort_pairs.append(specified_pair)
            else:
                expanded_keys = []
                for sort_key in specified_pair:
                    elements = []
                    for result_key in results.keys():
                        if result_key.startswith(sort_key + '.'):
                            elements.append(result_key)
                    expanded_keys.append(elements)

                sort_pairs.extend(list(zip(*expanded_keys)))
        return sort_pairs

    def map_sorting(results):
        ranking = create_2d_sort_matrix([results[map_name] for map_name in sort_specification.keys()],
                                        reversed_sort=True)
        list_index = np.arange(ranking.shape[0])
        sort_pairs = resolve_sort_pairs(results)

        updates = {}
        for names in sort_pairs:
            sort_matrix = np.column_stack([results[map_name] for map_name in names])
            sorted_maps = [sort_matrix[list_index, ranking[:, ind], None] for ind in range(ranking.shape[1])]
            updates.update(dict(zip(names, sorted_maps)))
        return updates

    return map_sorting


class DTIMeasures:

    @staticmethod
    def extra_optimization_maps(results):
        """Return some interesting measures like FA, MD, RD and AD.

        This function is meant to be used as a post processing routine in Tensor-like compartment models.

        Args:
            results (dict): Dictionary containing at least theta, phi, psi, d, dperp0 and dperp1
                We will use this to generate some standard measures from the diffusion Tensor.

        Returns:
            dict: as keys typical elements like 'FA and 'MD' as interesting output and as per values the maps.
                These maps are per voxel, and optionally per instance per voxel
        """
        output = {
            'FA': DTIMeasures.fractional_anisotropy(results['d'], results['dperp0'], results['dperp1']),
            'MD': (results['d'] + results['dperp0'] + results['dperp1']) / 3.,
            'AD': results['d'],
            'RD': (results['dperp0'] + results['dperp1']) / 2.0,
        }

        if all('{}.std'.format(el) in results for el in ['d', 'dperp0', 'dperp1']):
            output.update({
                'FA.std': DTIMeasures.fractional_anisotropy_std(
                    results['d'], results['dperp0'], results['dperp1'],
                    results['d.std'], results['dperp0.std'], results['dperp1.std'],
                    covariances=results.get('covariances', None)
                ),
                'MD.std': np.sqrt(results['d.std'] + results['dperp0.std'] + results['dperp1.std']) / 3.,
                'AD.std': results['d.std'],
                'RD.std': (results['dperp0.std'] + results['dperp1.std']) / 2.0,
            })

        if all(el in results for el in ['theta', 'phi', 'psi']):
            eigenvectors = tensor_spherical_to_cartesian(np.squeeze(results['theta']),
                                                         np.squeeze(results['phi']),
                                                         np.squeeze(results['psi']))
            for ind in range(3):
                output.update({'vec{}'.format(ind): eigenvectors[ind]})

        return output

    @staticmethod
    def extra_sampling_maps(results):
        """Return some interesting measures derived from the samples.

        Please note that this function expects the result dictionary only with the parameter names, that is,
        it expects the elements ``d``, ``dperp0`` and ``dperp1`` to be present.

        Args:
            results (dict[str: ndarray]): a dictionary containing the samples for each of the parameters.

        Returns:
            dict: a set of additional maps with one value per voxel.
        """
        items = [
            ('MD', (results['d'] + results['dperp0'] + results['dperp1']) / 3.),
            ('FA', DTIMeasures.fractional_anisotropy(results['d'], results['dperp0'], results['dperp1'])),
            ('RD', (results['dperp0'] + results['dperp1']) / 2.0),
            ('AD', results['d']),
            ('d', results['d']),
            ('dperp0', results['dperp0']),
            ('dperp1', results['dperp1']),
        ]

        results = {}
        for name, data in items:
            results.update({name: np.mean(data, axis=1),
                            name + '.std': np.std(data, axis=1)})
        return results

    @staticmethod
    def post_optimization_modifier(parameters_dict):
        """Apply post optimization modification of the Tensor compartment.

        This will re-orient the Tensor such that the eigenvalues are in decreasing order. This is done by
        permuting the eigen-values and -vectors and then recreating theta, phi and psi to match the rotated system.

        This is done primarily to be able to directly use the Tensor results in MCMC sample. Since we often put a
        prior on the diffusivities to be in decreasing order, we need to make sure that the starting point is valid.

        Args:
            parameters_dict (dict): the results from optimization. This expects each value to be a (n, ...) array with
                for each voxel either a scalar or a vector.

        Returns:
            dict: same set of parameters but then possibly updated with a rotation.
        """
        sorted_eigenvalues, sorted_eigenvectors, ranking = DTIMeasures._sort_eigensystem(parameters_dict)
        theta, phi, psi = tensor_cartesian_to_spherical(sorted_eigenvectors[0], sorted_eigenvectors[1])
        return {'d': sorted_eigenvalues[:, 0], 'dperp0': sorted_eigenvalues[:, 1], 'dperp1': sorted_eigenvalues[:, 2],
                'theta': theta, 'phi': phi, 'psi': psi}

    @staticmethod
    def fractional_anisotropy(d, dperp0, dperp1):
        """Calculate the fractional anisotropy (FA).

        Returns:
            ndarray: the fractional anisotropy for each voxel.
        """
        def compute(d, dperp0, dperp1):
            d, dperp0, dperp1 = map(lambda el: np.squeeze(el).astype(np.float64), [d, dperp0, dperp1])
            return np.sqrt(1 / 2.) * np.sqrt(((d - dperp0) ** 2 + (dperp0 - dperp1) ** 2 + (dperp1 - d) ** 2)
                                             / (d ** 2 + dperp0 ** 2 + dperp1 ** 2))

        if len(d.shape) > 1 and d.shape[1] > 1:
            fa = np.zeros(d.shape[:2])
            for batch_start, batch_end in split_in_batches(d.shape[1], 100):
                fa[:, batch_start:batch_end] = compute(
                    d[:, batch_start:batch_end],
                    dperp0[:, batch_start:batch_end],
                    dperp1[:, batch_start:batch_end])
            return fa
        else:
            return compute(d, dperp0, dperp1)

    @staticmethod
    def fractional_anisotropy_std(d, dperp0, dperp1, d_std, dperp0_std, dperp1_std, covariances=None):
        """Calculate the standard deviation of the fractional anisotropy (FA) using error propagation.

        Args:
            d (ndarray): an 1d array
            dperp0 (ndarray): an 1d array
            dperp1 (ndarray): an 1d array
            d_std (ndarray): an 1d array
            dperp0_std (ndarray): an 1d array
            dperp1_std (ndarray): an 1d array
            covariances (dict): optionally, a matrix holding the covariances. This expects the keys to be like:
                '<param_0>_to_<param_1>'. The order of the parameter names does not matter.

        Returns:
            ndarray: the standard deviation of the fraction anisotropy using error propagation of the diffusivities.
        """
        gradient = DTIMeasures._get_fractional_anisotropy_gradient(d, dperp0, dperp1)
        covars = create_covariance_matrix(
                {'d.std': d_std, 'dperp0.std': dperp0_std, 'dperp1.std': dperp1_std},
                ['d', 'dperp0', 'dperp1'], covariances)

        return np.nan_to_num(np.sqrt(voxelwise_vector_matrix_vector_product(gradient, covars, gradient)))

    @staticmethod
    def _get_fractional_anisotropy_gradient(d, dperp0, dperp1):
        """Get the gradient of the Fractional Anisotropy function.

        This returns the gradient of the Fractional Anisotropy (FA) function, evaluated at the given diffusivities.
        This is required for error propagating the uncertainties of the diffusivities into FA. The gradient is given
        by the partial derivative of:

        .. math::

            \text{FA} = \sqrt{\frac{1}{2}} \frac{\sqrt{(d - d_{\perp_0})^2 + (d_{\perp_0} - d_{\perp_1})^2
                        + (d_{\perp_1} - d)^2}}{\sqrt{d^2 + d_{\perp_0}^2 + d_{\perp_1}^2}}

        Args:
            d (ndarray): an 1d vector with the principal diffusivity per voxel
            dperp0 (ndarray): an 1d vector with the first perpendicular diffusivity per voxel
            dperp1 (ndarray): an 1d vector with the second perpendicular diffusivity per voxel

        Returns:
            ndarray: a 2d vector with the gradient per voxel.
        """
        np.warnings.simplefilter("ignore")

        d, dperp0, dperp1 = (np.squeeze(el).astype(np.float64) for el in [d, dperp0, dperp1])

        gradient = np.stack([
            (d ** 2 * (dperp0 + dperp1) + 2 * d * dperp0 * dperp1 - dperp0 ** 3
             - dperp0 ** 2 * dperp1 - dperp0 * dperp1 ** 2 - dperp1 ** 3)
            / (2 * (d ** 2 + dperp0 ** 2 + dperp1 ** 2) ** (3 / 2.)
               * np.sqrt(d ** 2 - d * (dperp0 + dperp1) + dperp0 ** 2 - dperp0 * dperp1 + dperp1 ** 2)),
            (-d ** 3 - d ** 2 * dperp1 + d * (dperp0 ** 2 + 2 * dperp0 * dperp1 - dperp1 ** 2) + dperp1 * (
            dperp0 ** 2 - dperp1 ** 2))
            / (2 * (d ** 2 + dperp0 ** 2 + dperp1 ** 2) ** (3 / 2.)
               * np.sqrt(d ** 2 - d * (dperp0 + dperp1) + dperp0 ** 2 - dperp0 * dperp1 + dperp1 ** 2)),
            (-d ** 3 - d ** 2 * dperp0 + d * (
            -dperp0 ** 2 + 2 * dperp0 * dperp1 + dperp1 ** 2) - dperp0 ** 3 + dperp0 * dperp1 ** 2)
            / (2 * (d ** 2 + dperp0 ** 2 + dperp1 ** 2) ** (3 / 2.)
               * np.sqrt(d ** 2 - d * (dperp0 + dperp1) + dperp0 ** 2 - dperp0 * dperp1 + dperp1 ** 2))
        ], axis=-1)

        if len(gradient.shape) < 2:
            return gradient[None, :]
        return gradient

    @staticmethod
    def _sort_eigensystem(parameters_dict):
        """Sort the eigensystem of the Tensor parameterized by eigen values and vectors.

        Args:
            parameters_dict (dict): the results from optimization. This expects each value to be a (n, ...) array with
                for each voxel either a scalar or a vector.

        Returns:
            tuple: the sorted eigenvalues,
        """
        eigenvectors = np.stack(tensor_spherical_to_cartesian(np.squeeze(parameters_dict['theta']),
                                                              np.squeeze(parameters_dict['phi']),
                                                              np.squeeze(parameters_dict['psi'])), axis=0)

        eigenvalues = np.atleast_2d(np.squeeze(np.dstack([parameters_dict['d'],
                                                          parameters_dict['dperp0'],
                                                          parameters_dict['dperp1']])))

        ranking = np.atleast_2d(np.squeeze(np.argsort(eigenvalues, axis=1, kind='mergesort')[:, ::-1]))
        voxels_range = np.arange(ranking.shape[0])
        sorted_eigenvalues = np.concatenate([eigenvalues[voxels_range, ranking[:, ind], None]
                                             for ind in range(ranking.shape[1])], axis=1)
        sorted_eigenvectors = np.stack([eigenvectors[ranking[:, ind], voxels_range, :]
                                        for ind in range(ranking.shape[1])])

        return sorted_eigenvalues, sorted_eigenvectors, ranking


class DKIMeasures:

    @staticmethod
    def extra_optimization_maps(parameters_dict):
        """Calculate DKI statistics like the mean, axial and radial kurtosis.

        The Mean Kurtosis (MK) is calculated by averaging the Kurtosis over orientations on the unit sphere.
        The Axial Kurtosis (AK) is obtained using the principal direction of diffusion (fe; first eigenvec)
        from the Tensor as its direction and then averaging the Kurtosis over +fe and -fe.
        Finally, the Radial Kurtosis (RK) is calculated by averaging the Kurtosis over a circle of directions around
        the first eigenvec.

        Args:
            parameters_dict (dict): the fitted Kurtosis parameters, this requires a dictionary with at least
                the elements:
                'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi', 'W_0000', 'W_1000', 'W_1100', 'W_1110',
                'W_1111', 'W_2000', 'W_2100', 'W_2110', 'W_2111', 'W_2200', 'W_2210', 'W_2211',
                'W_2220', 'W_2221', 'W_2222'.

        Returns:
            dict: maps for the Mean Kurtosis (MK), Axial Kurtosis (AK) and Radial Kurtosis (RK).
        """
        param_names = ['d', 'dperp0', 'dperp1', 'theta', 'phi', 'psi', 'W_0000', 'W_1000', 'W_1100', 'W_1110',
                       'W_1111', 'W_2000', 'W_2100', 'W_2110', 'W_2111', 'W_2200', 'W_2210', 'W_2211',
                       'W_2220', 'W_2221', 'W_2222']

        parameters = np.column_stack([parameters_dict[n] for n in param_names])

        nmr_voxels = parameters.shape[0]
        kernel_data = {'parameters': Array(parameters, ctype='mot_float_type'),
                       'directions': Array(DKIMeasures._get_spherical_samples(), ctype='float4',
                                           offset_str='0'),
                       'nmr_directions': Scalar(DKIMeasures._get_spherical_samples().shape[0]),
                       'nmr_radial_directions': Scalar(256),
                       'mks': Zeros((nmr_voxels,), ctype='float'),
                       'aks': Zeros((nmr_voxels,), ctype='float'),
                       'rks': Zeros((nmr_voxels,), ctype='float')}

        DKIMeasures._get_compute_function(param_names).evaluate(kernel_data, nmr_voxels)

        return {'MK': kernel_data['mks'].get_data(),
                'AK': kernel_data['aks'].get_data(),
                'RK': kernel_data['rks'].get_data()}

    @staticmethod
    def _get_compute_function(param_names):
        def get_param_cl_ref(param_name):
            return 'parameters[{}]'.format(param_names.index(param_name))

        param_expansions = ['mot_float_type {} = params[{}];'.format(name, ind) for ind, name in enumerate(param_names)]

        return parse_cl_function('''
            double apparent_kurtosis(
                    global mot_float_type* params,
                    float4 direction,
                    float4 vec0,
                    float4 vec1,
                    float4 vec2){

                ''' + '\n'.join(param_expansions) + '''

                double adc = d *      pown(dot(vec0, direction), 2) +
                             dperp0 * pown(dot(vec1, direction), 2) +
                             dperp1 * pown(dot(vec2, direction), 2);

                double tensor_md = (d + dperp0 + dperp1) / 3.0;

                double kurtosis_sum = KurtosisMultiplication(
                    W_0000, W_1111, W_2222, W_1000, W_2000, W_1110,
                    W_2220, W_2111, W_2221, W_1100, W_2200, W_2211,
                    W_2100, W_2110, W_2210, direction);

                return pown(tensor_md / adc, 2) * kurtosis_sum;
            }
        
            void get_principal_and_perpendicular_eigenvector(
                    mot_float_type d,
                    mot_float_type dperp0,
                    mot_float_type dperp1,
                    float4* vec0,
                    float4* vec1,
                    float4* vec2,
                    float4** principal_vec,
                    float4** perpendicular_vec){

                if(d >= dperp0 && d >= dperp1){
                    *principal_vec = vec0;
                    *perpendicular_vec = vec1;
                }
                if(dperp0 >= d && dperp0 >= dperp1){
                    *principal_vec = vec1;
                    *perpendicular_vec = vec0;
                }
                *principal_vec = vec2;
                *perpendicular_vec = vec0;
            }
        
            void calculate_measures(global mot_float_type* parameters, 
                                    global float4* directions,
                                    uint nmr_directions,
                                    uint nmr_radial_directions,
                                    global float* mks,
                                    global float* aks,
                                    global float* rks){
                int i, j;

                float4 vec0, vec1, vec2;
                TensorSphericalToCartesian(
                    ''' + get_param_cl_ref('theta') + ''',
                    ''' + get_param_cl_ref('phi') + ''',
                    ''' + get_param_cl_ref('psi') + ''',
                    &vec0, &vec1, &vec2);

                float4* principal_vec;
                float4* perpendicular_vec;
                get_principal_and_perpendicular_eigenvector(
                    ''' + get_param_cl_ref('d') + ''',
                    ''' + get_param_cl_ref('dperp0') + ''',
                    ''' + get_param_cl_ref('dperp1') + ''',
                    &vec0, &vec1, &vec2,
                    &principal_vec, &perpendicular_vec);

                // Mean Kurtosis integrated over a set of directions
                double mean = 0;
                for(i = 0; i < nmr_directions; i++){
                    mean += apparent_kurtosis(parameters, directions[i], vec0, vec1, vec2);
                }
                *(mks) = clamp(mean / nmr_directions, 0.0, 3.0);


                // Axial Kurtosis over the principal direction of diffusion
                *(aks) = clamp(apparent_kurtosis(parameters, *principal_vec, vec0, vec1, vec2), 0.0, 10.0);


                // Radial Kurtosis integrated over a unit circle around the principal eigenvector.
                mean = 0;
                float4 rotated_vec;
                for(i = 0; i < nmr_radial_directions; i++){
                    rotated_vec = RotateOrthogonalVector(*principal_vec, *perpendicular_vec,
                                                         i * (2 * M_PI_F) / nmr_radial_directions);

                    mean += (apparent_kurtosis(parameters, rotated_vec, vec0, vec1, vec2) - mean) / (i + 1);
                }
                *(rks) = max(mean, 0.0);
            }
        ''', dependencies=[get_component('library_functions', 'RotateOrthogonalVector')(),
                           get_component('library_functions', 'TensorSphericalToCartesian')(),
                           get_component('library_functions', 'KurtosisMultiplication')()])

    @staticmethod
    def _get_spherical_samples():
        """Get a number of 3d coordinates mapping an unit sphere.

        List taken from "dki_parameters.m" by Jelle Veraart
        (https://github.com/NYU-DiffusionMRI/Diffusion-Kurtosis-Imaging/blob/master/dki_parameters.m).

        Returns:
            ndarray: a list of 3d coordinates mapping an unit sphere.
        """
        return np.array([[0, 0, 1.0000],
                         [0.5924, 0, 0.8056],
                         [-0.7191, -0.1575, -0.6768],
                         [-0.9151, -0.3479, 0.2040],
                         [0.5535, 0.2437, 0.7964],
                         [-0.0844, 0.9609, -0.2636],
                         [0.9512, -0.3015, 0.0651],
                         [-0.4225, 0.8984, 0.1202],
                         [0.5916, -0.6396, 0.4909],
                         [0.3172, 0.8818, -0.3489],
                         [-0.1988, -0.6687, 0.7164],
                         [-0.2735, 0.3047, -0.9123],
                         [0.9714, -0.1171, 0.2066],
                         [-0.5215, -0.4013, 0.7530],
                         [-0.3978, -0.9131, -0.0897],
                         [0.2680, 0.8196, 0.5063],
                         [-0.6824, -0.6532, -0.3281],
                         [0.4748, -0.7261, -0.4973],
                         [0.4504, -0.4036, 0.7964],
                         [-0.5551, -0.8034, -0.2153],
                         [0.0455, -0.2169, 0.9751],
                         [0.0483, 0.5845, 0.8099],
                         [-0.1909, -0.1544, -0.9694],
                         [0.8383, 0.5084, 0.1969],
                         [-0.2464, 0.1148, 0.9623],
                         [-0.7458, 0.6318, 0.2114],
                         [-0.0080, -0.9831, -0.1828],
                         [-0.2630, 0.5386, -0.8005],
                         [-0.0507, 0.6425, -0.7646],
                         [0.4476, -0.8877, 0.1081],
                         [-0.5627, 0.7710, 0.2982],
                         [-0.3790, 0.7774, -0.5020],
                         [-0.6217, 0.4586, -0.6350],
                         [-0.1506, 0.8688, -0.4718],
                         [-0.4579, 0.2131, 0.8631],
                         [-0.8349, -0.2124, 0.5077],
                         [0.7682, -0.1732, -0.6163],
                         [0.0997, -0.7168, -0.6901],
                         [0.0386, -0.2146, -0.9759],
                         [0.9312, 0.1655, -0.3249],
                         [0.9151, 0.3053, 0.2634],
                         [0.8081, 0.5289, -0.2593],
                         [-0.3632, -0.9225, 0.1305],
                         [0.2709, -0.3327, -0.9033],
                         [-0.1942, -0.9790, -0.0623],
                         [0.6302, -0.7641, 0.1377],
                         [-0.6948, -0.3137, 0.6471],
                         [-0.6596, -0.6452, 0.3854],
                         [-0.9454, 0.2713, 0.1805],
                         [-0.2586, -0.7957, 0.5477],
                         [-0.3576, 0.6511, 0.6695],
                         [-0.8490, -0.5275, 0.0328],
                         [0.3830, 0.2499, -0.8893],
                         [0.8804, -0.2392, -0.4095],
                         [0.4321, -0.4475, -0.7829],
                         [-0.5821, -0.1656, 0.7961],
                         [0.3963, 0.6637, 0.6344],
                         [-0.7222, -0.6855, -0.0929],
                         [0.2130, -0.9650, -0.1527],
                         [0.4737, 0.7367, -0.4825],
                         [-0.9956, 0.0891, 0.0278],
                         [-0.5178, 0.7899, -0.3287],
                         [-0.8906, 0.1431, -0.4317],
                         [0.2431, -0.9670, 0.0764],
                         [-0.6812, -0.3807, -0.6254],
                         [-0.1091, -0.5141, 0.8507],
                         [-0.2206, 0.7274, -0.6498],
                         [0.8359, 0.2674, 0.4794],
                         [0.9873, 0.1103, 0.1147],
                         [0.7471, 0.0659, -0.6615],
                         [0.6119, -0.2508, 0.7502],
                         [-0.6191, 0.0776, 0.7815],
                         [0.7663, -0.4739, 0.4339],
                         [-0.5699, 0.5369, 0.6220],
                         [0.0232, -0.9989, 0.0401],
                         [0.0671, -0.4207, -0.9047],
                         [-0.2145, 0.5538, 0.8045],
                         [0.8554, -0.4894, 0.1698],
                         [-0.7912, -0.4194, 0.4450],
                         [-0.2341, 0.0754, -0.9693],
                         [-0.7725, 0.6346, -0.0216],
                         [0.0228, 0.7946, -0.6067],
                         [0.7461, -0.3966, -0.5348],
                         [-0.4045, -0.0837, -0.9107],
                         [-0.4364, 0.6084, -0.6629],
                         [0.6177, -0.3175, -0.7195],
                         [-0.4301, -0.0198, 0.9026],
                         [-0.1489, -0.9706, 0.1892],
                         [0.0879, 0.9070, -0.4117],
                         [-0.7764, -0.4707, -0.4190],
                         [0.9850, 0.1352, -0.1073],
                         [-0.1581, -0.3154, 0.9357],
                         [0.8938, -0.3246, 0.3096],
                         [0.8358, -0.4464, -0.3197],
                         [0.4943, 0.4679, 0.7327],
                         [-0.3095, 0.9015, -0.3024],
                         [-0.3363, -0.8942, -0.2956],
                         [-0.1271, -0.9274, -0.3519],
                         [0.3523, -0.8717, -0.3407],
                         [0.7188, -0.6321, 0.2895],
                         [-0.7447, 0.0924, -0.6610],
                         [0.1622, 0.7186, 0.6762],
                         [-0.9406, -0.0829, -0.3293],
                         [-0.1229, 0.9204, 0.3712],
                         [-0.8802, 0.4668, 0.0856],
                         [-0.2062, -0.1035, 0.9730],
                         [-0.4861, -0.7586, -0.4338],
                         [-0.6138, 0.7851, 0.0827],
                         [0.8476, 0.0504, 0.5282],
                         [0.3236, 0.4698, -0.8213],
                         [-0.7053, -0.6935, 0.1473],
                         [0.1511, 0.3778, 0.9135],
                         [0.6011, 0.5847, 0.5448],
                         [0.3610, 0.3183, 0.8766],
                         [0.9432, 0.3304, 0.0341],
                         [0.2423, -0.8079, -0.5372],
                         [0.4431, -0.1578, 0.8825],
                         [0.6204, 0.5320, -0.5763],
                         [-0.2806, -0.5376, -0.7952],
                         [-0.5279, -0.8071, 0.2646],
                         [-0.4214, -0.6159, 0.6656],
                         [0.6759, -0.5995, -0.4288],
                         [0.5670, 0.8232, -0.0295],
                         [-0.0874, 0.4284, -0.8994],
                         [0.8780, -0.0192, -0.4782],
                         [0.0166, 0.8421, 0.5391],
                         [-0.7741, 0.2931, -0.5610],
                         [0.9636, -0.0579, -0.2611],
                         [0, 0, -1.0000],
                         [-0.5924, 0, -0.8056],
                         [0.7191, 0.1575, 0.6768],
                         [0.9151, 0.3479, -0.2040],
                         [-0.5535, -0.2437, -0.7964],
                         [0.0844, -0.9609, 0.2636],
                         [-0.9512, 0.3015, -0.0651],
                         [0.4225, -0.8984, -0.1202],
                         [-0.5916, 0.6396, -0.4909],
                         [-0.3172, -0.8818, 0.3489],
                         [0.1988, 0.6687, -0.7164],
                         [0.2735, -0.3047, 0.9123],
                         [-0.9714, 0.1171, -0.2066],
                         [0.5215, 0.4013, -0.7530],
                         [0.3978, 0.9131, 0.0897],
                         [-0.2680, -0.8196, -0.5063],
                         [0.6824, 0.6532, 0.3281],
                         [-0.4748, 0.7261, 0.4973],
                         [-0.4504, 0.4036, -0.7964],
                         [0.5551, 0.8034, 0.2153],
                         [-0.0455, 0.2169, -0.9751],
                         [-0.0483, -0.5845, -0.8099],
                         [0.1909, 0.1544, 0.9694],
                         [-0.8383, -0.5084, -0.1969],
                         [0.2464, -0.1148, -0.9623],
                         [0.7458, -0.6318, -0.2114],
                         [0.0080, 0.9831, 0.1828],
                         [0.2630, -0.5386, 0.8005],
                         [0.0507, -0.6425, 0.7646],
                         [-0.4476, 0.8877, -0.1081],
                         [0.5627, -0.7710, -0.2982],
                         [0.3790, -0.7774, 0.5020],
                         [0.6217, -0.4586, 0.6350],
                         [0.1506, -0.8688, 0.4718],
                         [0.4579, -0.2131, -0.8631],
                         [0.8349, 0.2124, -0.5077],
                         [-0.7682, 0.1732, 0.6163],
                         [-0.0997, 0.7168, 0.6901],
                         [-0.0386, 0.2146, 0.9759],
                         [-0.9312, -0.1655, 0.3249],
                         [-0.9151, -0.3053, -0.2634],
                         [-0.8081, -0.5289, 0.2593],
                         [0.3632, 0.9225, -0.1305],
                         [-0.2709, 0.3327, 0.9033],
                         [0.1942, 0.9790, 0.0623],
                         [-0.6302, 0.7641, -0.1377],
                         [0.6948, 0.3137, -0.6471],
                         [0.6596, 0.6452, -0.3854],
                         [0.9454, -0.2713, -0.1805],
                         [0.2586, 0.7957, -0.5477],
                         [0.3576, -0.6511, -0.6695],
                         [0.8490, 0.5275, -0.0328],
                         [-0.3830, -0.2499, 0.8893],
                         [-0.8804, 0.2392, 0.4095],
                         [-0.4321, 0.4475, 0.7829],
                         [0.5821, 0.1656, -0.7961],
                         [-0.3963, -0.6637, -0.6344],
                         [0.7222, 0.6855, 0.0929],
                         [-0.2130, 0.9650, 0.1527],
                         [-0.4737, -0.7367, 0.4825],
                         [0.9956, -0.0891, -0.0278],
                         [0.5178, -0.7899, 0.3287],
                         [0.8906, -0.1431, 0.4317],
                         [-0.2431, 0.9670, -0.0764],
                         [0.6812, 0.3807, 0.6254],
                         [0.1091, 0.5141, -0.8507],
                         [0.2206, -0.7274, 0.6498],
                         [-0.8359, -0.2674, -0.4794],
                         [-0.9873, -0.1103, -0.1147],
                         [-0.7471, -0.0659, 0.6615],
                         [-0.6119, 0.2508, -0.7502],
                         [0.6191, -0.0776, -0.7815],
                         [-0.7663, 0.4739, -0.4339],
                         [0.5699, -0.5369, -0.6220],
                         [-0.0232, 0.9989, -0.0401],
                         [-0.0671, 0.4207, 0.9047],
                         [0.2145, -0.5538, -0.8045],
                         [-0.8554, 0.4894, -0.1698],
                         [0.7912, 0.4194, -0.4450],
                         [0.2341, -0.0754, 0.9693],
                         [0.7725, -0.6346, 0.0216],
                         [-0.0228, -0.7946, 0.6067],
                         [-0.7461, 0.3966, 0.5348],
                         [0.4045, 0.0837, 0.9107],
                         [0.4364, -0.6084, 0.6629],
                         [-0.6177, 0.3175, 0.7195],
                         [0.4301, 0.0198, -0.9026],
                         [0.1489, 0.9706, -0.1892],
                         [-0.0879, -0.9070, 0.4117],
                         [0.7764, 0.4707, 0.4190],
                         [-0.9850, -0.1352, 0.1073],
                         [0.1581, 0.3154, -0.9357],
                         [-0.8938, 0.3246, -0.3096],
                         [-0.8358, 0.4464, 0.3197],
                         [-0.4943, -0.4679, -0.7327],
                         [0.3095, -0.9015, 0.3024],
                         [0.3363, 0.8942, 0.2956],
                         [0.1271, 0.9274, 0.3519],
                         [-0.3523, 0.8717, 0.3407],
                         [-0.7188, 0.6321, -0.2895],
                         [0.7447, -0.0924, 0.6610],
                         [-0.1622, -0.7186, -0.6762],
                         [0.9406, 0.0829, 0.3293],
                         [0.1229, -0.9204, -0.3712],
                         [0.8802, -0.4668, -0.0856],
                         [0.2062, 0.1035, -0.9730],
                         [0.4861, 0.7586, 0.4338],
                         [0.6138, -0.7851, -0.0827],
                         [-0.8476, -0.0504, -0.5282],
                         [-0.3236, -0.4698, 0.8213],
                         [0.7053, 0.6935, -0.1473],
                         [-0.1511, -0.3778, -0.9135],
                         [-0.6011, -0.5847, -0.5448],
                         [-0.3610, -0.3183, -0.8766],
                         [-0.9432, -0.3304, -0.0341],
                         [-0.2423, 0.8079, 0.5372],
                         [-0.4431, 0.1578, -0.8825],
                         [-0.6204, -0.5320, 0.5763],
                         [0.2806, 0.5376, 0.7952],
                         [0.5279, 0.8071, -0.2646],
                         [0.4214, 0.6159, -0.6656],
                         [-0.6759, 0.5995, 0.4288],
                         [-0.5670, -0.8232, 0.0295],
                         [0.0874, -0.4284, 0.8994],
                         [-0.8780, 0.0192, 0.4782],
                         [-0.0166, -0.8421, -0.5391],
                         [0.7741, -0.2931, 0.5610],
                         [-0.9636, 0.0579, 0.2611]])


class NODDIMeasures:

    @staticmethod
    def noddi_watson_extra_optimization_maps(results):
        """Computes the NDI and ODI for the NODDI Watson model"""
        return {'NDI': results['w_ic.w'] / (results['w_ic.w'] + results['w_ec.w']),
                'ODI': np.arctan2(1.0, results['NODDI_IC.kappa']) * 2 / np.pi}

    @staticmethod
    def noddi_watson_extra_sampling_maps(results):
        """Computes the NDI and ODI per sample and average over the derived values."""
        ndi = results['w_ic.w'] / (results['w_ic.w'] + results['w_ec.w'])
        odi = np.arctan2(1.0, results['NODDI_IC.kappa']) * 2 / np.pi

        return {'NDI': np.mean(ndi, axis=1), 'NDI.std': np.std(ndi, axis=1),
                'ODI': np.mean(odi, axis=1), 'ODI.std': np.std(odi, axis=1)}

    @staticmethod
    def noddi_bingham_extra_optimization_maps(results):
        """Computes the ODI's and Dispersion Anisotropic Index (DAI) for the NODDI Bingham model"""
        def compute(ind, kappa, beta):
            return {'ODI_p{}'.format(ind): np.arctan2(1.0, kappa - beta) * 2 / np.pi,
                    'ODI_s{}'.format(ind): np.arctan2(1.0, kappa) * 2 / np.pi,
                    'ODI{}'.format(ind): np.arctan2(1.0, np.sqrt(np.abs(kappa * (kappa - beta)))) * 2 / np.pi,
                    'DAI{}'.format(ind): np.arctan2(beta, kappa - beta) * 2 / np.pi}

        output = {}
        for ind in range(2):
            if 'BinghamNODDI_IN{}.k1'.format(ind) in results:
                output.update(compute(
                    ind,
                    results['BinghamNODDI_IN{}.k1'.format(ind)],
                    results['BinghamNODDI_IN{}.k1'.format(ind)] / results['BinghamNODDI_IN{}.kw'.format(ind)]))
        return output

    @staticmethod
    def noddi_bingham_extra_sampling_maps(results):
        """Computes the ODI's and Dispersion Anisotropic Index (DAI) for the NODDI Bingham model.

        This computes the indices per sample and takes the mean and std. over that.
        """
        def compute(ind, kappa, beta):
            odi_p = np.arctan2(1.0, kappa - beta) * 2 / np.pi
            odi_s = np.arctan2(1.0, kappa) * 2 / np.pi
            odi = np.arctan2(1.0, np.sqrt(np.abs(kappa * (kappa - beta)))) * 2 / np.pi
            dai = np.arctan2(beta, kappa - beta) * 2 / np.pi

            return {'ODI_p{}'.format(ind): np.mean(odi_p, axis=1),
                    'ODI_p{}.std'.format(ind): np.std(odi_p, axis=1),
                    'ODI_s{}'.format(ind): np.mean(odi_s, axis=1),
                    'ODI_s{}.std'.format(ind): np.std(odi_s, axis=1),
                    'ODI{}'.format(ind): np.mean(odi, axis=1),
                    'ODI{}.std'.format(ind): np.std(odi, axis=1),
                    'DAI{}'.format(ind): np.mean(dai, axis=1),
                    'DAI{}.std'.format(ind): np.std(dai, axis=1)}

        output = {}
        for ind in range(2):
            if 'BinghamNODDI_IN{}.k1'.format(ind) in results:
                output.update(compute(
                    ind,
                    results['BinghamNODDI_IN{}.k1'.format(ind)],
                    results['BinghamNODDI_IN{}.k1'.format(ind)] / results['BinghamNODDI_IN{}.kw'.format(ind)]))
        return output


def noddi_dti_maps(results):
    """Compute NODDI-like statistics from Tensor/Kurtosis parameter fits.

    Several authors noted correspondence between NODDI parameters and DTI parameters [1, 2]. This function computes
    the neurite density index (NDI) and NODDI's measure of neurite dispersion using Tensor parameters.

    The corresponding theory assumes that the intrinsic diffusivity of the intra-neurite compartment of NODDI
    is fixed to d = 1.7 x 10^-9 m^2 s^-1. As such, we fix it here to that value as well and compute the corresponding
    NODDI-DTI results.

    Args:
        results (mdt.models.composite.ExtraOptimizationMapsInfo): the results data, should contain at least:

            - d (ndarray): principal diffusivity
            - dperp0 (ndarray): primary perpendicular diffusion
            - dperp1 (ndarray): primary perpendicular diffusion

            And, if present, we also use these:

            - FA (ndarray): if computed already, the Fractional Anisotropy of the given diffusivities
            - MD (ndarray): if computed already, the Mean Diffusivity of the given diffusivities
            - MK (ndarray): if computing for Kurtosis, the computed Mean Kurtosis. If not given, we assume unity.

    Returns:
        dict: maps for the the NODDI-DTI, NDI and ODI measures.

    References:
        1. Edwards LJ, Pine KJ, Ellerbrock I, Weiskopf N, Mohammadi S. NODDI-DTI: Estimating neurite orientation and
            dispersion parameters from a diffusion tensor in healthy white matter.
            Front Neurosci. 2017;11(DEC):1-15. doi:10.3389/fnins.2017.00720.
        2. Lampinen B, Szczepankiewicz F, Martensson J, van Westen D, Sundgren PC, Nilsson M. Neurite density
            imaging versus imaging of microscopic anisotropy in diffusion MRI: A model comparison using spherical
            tensor encoding. Neuroimage. 2017;147(July 2016):517-531. doi:10.1016/j.neuroimage.2016.11.053.
    """
    noddi_d = 1.7e-9

    d = results['d']
    dperp0 = results['dperp0']
    dperp1 = results['dperp1']

    FA = results.get('FA', DTIMeasures.fractional_anisotropy(d, dperp0, dperp1))
    MD = results.get('MD', (d + dperp0 + dperp1) / 3.)

    tau = 1 / 3. * (1 + (4 / np.fabs(noddi_d - MD)) * (MD * FA / np.sqrt(3 - 2 * FA ** 2)))
    tau = np.nan_to_num(tau)

    kappa = _tau_to_kappa(tau)
    odi = np.mean(np.arctan2(1.0, kappa) * 2 / np.pi, axis=1)

    shells = results.input_data.protocol.get_b_values_shells()
    if shells:
        b = shells[0]['b_value']
        if len(shells) > 1:
            b = shells[1]['b_value'] - shells[0]['b_value']

        sum = (d ** 2 + dperp0 ** 2 + dperp1 ** 2) / 5 + 2 * (d * dperp0 + d * dperp1 + dperp0 * dperp1) / 15

        MD += ((b / 6) * sum) * results.get('MK', 1)

    ndi = 1 - np.sqrt(0.5 * ((3 * MD) / noddi_d - 1))
    ndi = np.clip(np.nan_to_num(ndi), 0, 1)

    return {'NODDI_DTI_NDI': ndi, 'NODDI_DTI_TAU': tau, 'NODDI_DTI_KAPPA': kappa, 'NODDI_DTI_ODI': odi}


def _tau_to_kappa(tau):
    """Using non-linear optimization, convert the NODDI-DTI Tau variables to NODDI kappa's.

    Args:
        tau (ndarray): the list of tau's per voxel.

    Returns:
        ndarray: the list of corresponding kappa's
    """
    tau_func = SimpleCLFunction.from_string('''
        double tau(double kappa){
            if(kappa < 1e-12){
                return 1/3.0;
            }
            return 0.5 * ( 1 / ( sqrt(kappa) * dawson(sqrt(kappa) ) ) - 1/kappa);
        }''', dependencies=[dawson()])

    objective_func = SimpleCLFunction.from_string('''
        double tau_to_kappa(local const mot_float_type* const x, void* data, local mot_float_type* objective_list){
            return pown(tau(x[0]) - ((_tau_to_kappa_data*)data)->tau, 2); 
        }
    ''', dependencies=[tau_func])

    kappa = minimize(objective_func, np.ones_like(tau),
                     data=Struct({'tau': Array(tau, 'mot_float_type', as_scalar=True)},
                                 '_tau_to_kappa_data')).x
    kappa[kappa > 64] = 1
    kappa[kappa < 0] = 1
    return kappa

