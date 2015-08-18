from unittest import TestCase
import os

import numpy as np
from numpy.testing import assert_allclose
import pyopencl as cl

from mdt.library.General import CLEnvironmentFactory
from mot.load_balance_strategies import PreferGPU
from mot.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TestMultiLM(TestCase):

    def test_minimise(self):
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

        cl_func = open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'ExpT2Dec.cl')), 'r').read()
        cl_func += """
            double evaluateModel(optimize_data* data, double* x, const int current_voxel){
                    return (x[0] * cmExpT2Dec(data->prtcl_data_protocol[current_voxel], x[1]));
            }
        """

        cl_decode_func = """
            void decodeParameters(double* x){
                x[0] = fabs(x[0]);
                x[1] = fabs(x[1]);
            }
        """
        get_observation_func = '''
            double getObservation(optimize_data* data, const int current_voxel){
                    return data->var_data_signals[current_voxel];
            }
        '''
        protocol = np.array([[0.04], [0.045], [0.05], [0.06], [0.07], [0.08],
                       [0.09], [0.1], [0.11], [0.12], [0.13], [0.14]])
        voxels = np.array([[8744.47851562500, 7775.69433593750, 6826.66992187500, 6353.67968750000, 5963.06201171875,
                            5749.62841796875, 5722.99072265625, 5146.02392578125, 4829.35644531250, 4350.44042968750,
                            3912.88842773438, 4794.94921875000],
                           [8535.78320312500, 8134.85644531250, 6979.14208984375, 6552.66748046875, 6408.75146484375,
                            6669.22021484375, 6075.10205078125, 5902.21386718750, 5499.46679687500, 5192.56396484375,
                            4474.27392578125, 4980.79443359375]])
        starting_points = np.array([[1e5, 0.05],
                                    [1e5, 0.05]])

        # from matlab: >> voxels = signal4d_extra(45,44:45,36,:)

        cpu_environment_list = CLEnvironmentFactory.all_devices(
            cl_device_type=cl.device_type.CPU, compile_flags=('-cl-strict-aliasing',))

        lm = LevenbergMarquardt()
        result = lm._minimize(cl_func, get_observation_func, starting_points, {'signals': voxels},
                              {'protocol': protocol}, {}, voxels.shape[1], 2, 2, cpu_environment_list, PreferGPU(),
                              cl_decode_func)

        result = np.abs(result)

        expected = np.array([[10379.5393678579, 0.143661626154132],
                             [9917.54701981713, 0.182548645222656]])

        assert_allclose(result, expected, rtol=1e-4)