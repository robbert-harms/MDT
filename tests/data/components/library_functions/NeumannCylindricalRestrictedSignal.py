import unittest

from scipy.special import jnp_zeros
import numpy as np
import pyopencl as cl
from numpy.testing import assert_allclose

import mdt
from mot.utils import get_float_type_def


class test_NeumannCylindricalRestrictedSignal(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(test_NeumannCylindricalRestrictedSignal, self).__init__(*args, **kwargs)

    def test_ncrs_float(self):
        test_params = self._generate_test_params().astype(dtype=np.float32)

        python_results = self._calculate_python(test_params)
        cl_results = self._calculate_cl(test_params, double_precision=False)

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-5, rtol=1e-5)

    def test_ncrs_double(self):
        test_params = self._generate_test_params().astype(dtype=np.float64)

        python_results = self._calculate_python(test_params)
        cl_results = self._calculate_cl(test_params, double_precision=True)

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-7)

    def _calculate_cl(self, test_params, double_precision=False):
        if double_precision:
            test_params = test_params.astype(np.float64)
        else:
            test_params = test_params.astype(np.float32)

        src = self._get_kernel_source(double_precision, test_params.shape[0])
        results = np.zeros(test_params.shape[0])
        self._run_kernel(src, test_params, results)
        return results

    def _run_kernel(self, src, input_args, results):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, src).build()

        mf = cl.mem_flags
        input_args_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_args)
        output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=results)
        buffers = [input_args_buffer, output_buffer]

        kernel_event = prg.test_ncrs(queue, (input_args.shape[0],), None, *buffers)
        self._enqueue_readout(queue, output_buffer, results, 0, input_args.shape[0], [kernel_event])

    def _enqueue_readout(self, queue, buffer, host_array, range_start, range_end, wait_for):
        nmr_problems = range_end - range_start
        return cl.enqueue_map_buffer(
            queue, buffer, cl.map_flags.READ, range_start * host_array.strides[0],
            (nmr_problems, ) + host_array.shape[1:], host_array.dtype, order="C", wait_for=wait_for,
            is_blocking=False)[1]

    def _get_kernel_source(self, double_precision, nmr_problems):
        src = ''
        src += get_float_type_def(double_precision)
        src += mdt.get_component_class('library_functions', 'NeumannCylindricalRestrictedSignal')().get_cl_code()
        src += '''
            __kernel void test_ncrs(global mot_float_type input_args[''' + str(nmr_problems) + '''][5],
                                    global double output[''' + str(nmr_problems) + ''']){

                uint gid = get_global_id(0);
                
                mot_float_type Delta = input_args[gid][0];
                mot_float_type delta = input_args[gid][1];
                mot_float_type d = input_args[gid][2];
                mot_float_type R = input_args[gid][3];
                mot_float_type G = input_args[gid][4];
                
                output[gid] = NeumannCylindricalRestrictedSignal(Delta, delta, d, R, G);
            }
        '''
        return src

    def _generate_test_params(self):
        """

        [Delta (s), delta (s), d (m/s^2), R (m), G (T/m)]

        """

        test_param_sets = [
            {'default': [0.5, 1e-2, 1e-9, 1e-6, 0.05],
             'lower_bounds': [0.1, 1e-3, 1e-10, 1e-7, 1e-4],
             'upper_bounds': [1, 0.1, 1e-8, 2e-5, 0.1]}
        ]

        def generate_params_matrix(defaults, lower_bounds, upper_bounds, nmr_steps):
            params_matrix = np.tile(default_values, (nmr_steps * len(default_values), 1))

            for ind in range(len(default_values)):
                params_matrix[(ind * nmr_steps):((ind + 1) * nmr_steps), ind] = \
                    np.linspace(lower_bounds[ind], upper_bounds[ind], num=nmr_steps)

            return params_matrix

        nmr_steps = 100

        matrices = []

        for param_set in test_param_sets:
            default_values = param_set['default']
            lower_bounds = param_set['lower_bounds']
            upper_bounds = param_set['upper_bounds']

            matrices.append(generate_params_matrix(default_values, lower_bounds, upper_bounds, nmr_steps))

        test_cases = np.vstack(matrices)

        return test_cases

    def _calculate_python(self, input_params):
        results = np.zeros(input_params.shape[0])

        for ind in range(input_params.shape[0]):
            results[ind] = self._ncrs_python(*list(input_params[ind, :]))

        return results

    def _ncrs_python(self, Delta, delta, d, R, G):
        if R == 0 or R < np.finfo(float).eps:
            return 0

        GAMMA = 267.5987E6
        alpha_roots = jnp_zeros(1, 20) / R

        sum = 0
        for i in range(20):
            alpha = alpha_roots[i]

            num = (2 * d * alpha**2 * delta
                   - 2
                   + 2 * np.exp(-d * alpha**2 * delta)
                   + 2 * np.exp(-d * alpha**2 * Delta)
                   - np.exp(-d * alpha**2 * (Delta - delta))
                   - np.exp(-d * alpha**2 * (Delta + delta)))
            dem = d**2 * alpha**6 * (R**2 * alpha**2 - 1)

            sum += (num / dem)

        return -2 * GAMMA**2 * G**2 * sum

