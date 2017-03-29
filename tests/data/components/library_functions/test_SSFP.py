import unittest

import numpy as np
import pyopencl as cl
import mdt
from mot.utils import get_float_type_def


class test_SSFP(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(test_SSFP, self).__init__(*args, **kwargs)
        self.cases = [
            [(1e-9, 0, 10, 0.05, 1, 1, 1, 1), 100],
            [(1e-10, 0, 10, 0.05, 1, 1, 1, 1), 100],
            [(1e-10, 0, 10, 0.05, 1, 1, 1, 1), 100],
        ]

    def test_ssfp_float(self):
        src = self._get_kernel_source(False)

        input_args = np.array([el[0] for el in self.cases], dtype=np.float32)
        expected_results = np.array([el[1] for el in self.cases], dtype=np.float64)
        results = np.zeros_like(expected_results)

        self._run_kernel(src, input_args, results)
        print(results)

    def test_ssfp_double(self):
        src = self._get_kernel_source(True)

        input_args = np.array([el[0] for el in self.cases], dtype=np.float64)
        expected_results = np.array([el[1] for el in self.cases], dtype=np.float64)
        results = np.zeros_like(expected_results)

        self._run_kernel(src, input_args, results)
        print(results)

    def _run_kernel(self, src, input_args, results):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, src).build()

        mf = cl.mem_flags
        input_args_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_args)
        output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=results)
        buffers = [input_args_buffer, output_buffer]

        kernel_event = prg.test_ssfp(queue, (input_args.shape[0],), None, *buffers)
        self._enqueue_readout(queue, output_buffer, results, 0, input_args.shape[0], [kernel_event])

    def _enqueue_readout(self, queue, buffer, host_array, range_start, range_end, wait_for):
        nmr_problems = range_end - range_start
        return cl.enqueue_map_buffer(
            queue, buffer, cl.map_flags.READ, range_start * host_array.strides[0],
            (nmr_problems, ) + host_array.shape[1:], host_array.dtype, order="C", wait_for=wait_for,
            is_blocking=False)[1]

    def _get_kernel_source(self, double_precision):
        nmr_problems = len(self.cases)

        src = ''
        src += get_float_type_def(double_precision)
        src += mdt.get_component_class('library_functions', 'SSFP')().get_cl_code()
        src += '''
            __kernel void test_ssfp(global mot_float_type input_args[''' + str(nmr_problems) + '''][8],
                                    global double output[''' + str(nmr_problems) + ''']){

                uint gid = get_global_id(0);

                mot_float_type d = input_args[gid][0];
                mot_float_type delta = input_args[gid][1];
                mot_float_type G = input_args[gid][2];
                mot_float_type TR = input_args[gid][3];
                mot_float_type flip_angle = input_args[gid][4];
                mot_float_type b1 = input_args[gid][5];
                mot_float_type T1 = input_args[gid][6];
                mot_float_type T2 = input_args[gid][7];

                output[gid] = SSFP(d, delta, G, TR, flip_angle, b1, T1, T2);
            }
        '''
        return src
