import numpy as np
from mot.cl_routines.mapping.run_procedure import RunProcedure
from mot.kernel_data import KernelArray, KernelAllocatedArray, KernelScalar
from mot.cl_routines.base import CLRoutine
from mdt.components import get_component
from mot.utils import NameFunctionTuple

__author__ = 'Robbert Harms'
__date__ = "2017-08-16"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DKIMeasures(CLRoutine):

    def calculate(self, parameters_dict):
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

        parameters = np.require(np.column_stack([parameters_dict[n] for n in param_names]),
                                self._cl_runtime_info.mot_float_dtype, requirements=['C', 'A', 'O'])

        nmr_voxels = parameters.shape[0]
        kernel_data = {'parameters': KernelArray(parameters, ctype='mot_float_type',
                                                 is_readable=True, is_writable=False),
                       'directions': KernelArray(self._get_spherical_samples(), ctype='mot_float_type4',
                                                 is_readable=True, is_writable=False, offset_str='0'),
                       'nmr_directions': KernelScalar(self._get_spherical_samples().shape[0]),
                       'nmr_radial_directions': KernelScalar(256),
                       'mks': KernelAllocatedArray((nmr_voxels, 1), ctype='mot_float_type'),
                       'aks': KernelAllocatedArray((nmr_voxels, 1), ctype='mot_float_type'),
                       'rks': KernelAllocatedArray((nmr_voxels, 1), ctype='mot_float_type')}

        runner = RunProcedure(self._cl_runtime_info)
        runner.run_procedure(self._get_compute_function(param_names), kernel_data, nmr_voxels)

        return {'MK': kernel_data['mks'].get_data(),
                'AK': kernel_data['aks'].get_data(),
                'RK': kernel_data['rks'].get_data()}

    def _get_compute_function(self, param_names):
        def get_param_cl_ref(param_name):
            return 'data->parameters[{}]'.format(param_names.index(param_name))

        param_expansions = ['mot_float_type {} = params[{}];'.format(name, ind) for ind, name in enumerate(param_names)]

        kernel_source = ''
        kernel_source += get_component('library_functions', 'TensorApparentDiffusion')().get_cl_code()
        kernel_source += get_component('library_functions', 'RotateOrthogonalVector')().get_cl_code()
        kernel_source += get_component('library_functions', 'KurtosisMultiplication')().get_cl_code()
        kernel_source += '''
            double apparent_kurtosis(
                    global mot_float_type* params,
                    mot_float_type4 direction,
                    mot_float_type4 vec0,
                    mot_float_type4 vec1,
                    mot_float_type4 vec2){

                ''' + '\n'.join(param_expansions) + '''

                mot_float_type adc = d *      pown(dot(vec0, direction), 2) +
                                     dperp0 * pown(dot(vec1, direction), 2) +
                                     dperp1 * pown(dot(vec2, direction), 2);

                mot_float_type tensor_md = (d + dperp0 + dperp1) / 3.0;

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
                    mot_float_type4* vec0,
                    mot_float_type4* vec1,
                    mot_float_type4* vec2,
                    mot_float_type4** principal_vec,
                    mot_float_type4** perpendicular_vec){

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

            void calculate_measures(mot_data_struct* data){
                int i, j;

                mot_float_type4 vec0, vec1, vec2;
                TensorSphericalToCartesian(
                    ''' + get_param_cl_ref('theta') + ''',
                    ''' + get_param_cl_ref('phi') + ''',
                    ''' + get_param_cl_ref('psi') + ''',
                    &vec0, &vec1, &vec2);

                mot_float_type4* principal_vec;
                mot_float_type4* perpendicular_vec;
                get_principal_and_perpendicular_eigenvector(
                    ''' + get_param_cl_ref('d') + ''',
                    ''' + get_param_cl_ref('dperp0') + ''',
                    ''' + get_param_cl_ref('dperp1') + ''',
                    &vec0, &vec1, &vec2,
                    &principal_vec, &perpendicular_vec);


                // Mean Kurtosis integrated over a set of directions
                double mean = 0;
                for(i = 0; i < data->nmr_directions; i++){
                    mean += (apparent_kurtosis(data->parameters, data->directions[i], vec0, vec1, vec2) - mean) 
                                / (i + 1);
                }
                *(data->mks) = clamp(mean, (double)0, (double)3);


                // Axial Kurtosis over the principal direction of diffusion
                *(data->aks) = clamp(apparent_kurtosis(data->parameters, *principal_vec, vec0, vec1, vec2), 
                                   (double)0, (double)10);


                // Radial Kurtosis integrated over a unit circle around the principal eigenvector.
                mean = 0;
                mot_float_type4 rotated_vec;
                for(i = 0; i < data->nmr_radial_directions; i++){
                    rotated_vec = RotateOrthogonalVector(*principal_vec, *perpendicular_vec,
                                                         i * (2 * M_PI_F) / data->nmr_radial_directions);

                    mean += (apparent_kurtosis(data->parameters, rotated_vec, vec0, vec1, vec2) - mean) / (i + 1);
                }
                *(data->rks) = max(mean, (double)0);
            }
        '''
        return NameFunctionTuple('calculate_measures', kernel_source)

    def _get_spherical_samples(self):
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
