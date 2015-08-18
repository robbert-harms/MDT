from unittest import TestCase

import os
import pyopencl as cl
import numpy as np
from numpy.testing import assert_allclose
import scipy

from mdt.library.General import get_cl_double_extension_definer, get_bessel_roots
from mot.cl_functions import LMMin


__author__ = 'Robbert Harms'
__date__ = "2014-05-21"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TestOpenCL(TestCase):

    def setUp(self):
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        self.platform = cl.get_platforms()[0]

        devices = self.platform.get_devices()
        try:
            devices = self.platform.get_devices(device_type=cl.device_type.GPU)
        except cl.RuntimeError:
            pass

        self.device = devices[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

    def test_compModel_Ball(self):
        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'Ball.cl')), 'r').read()
        kernel_source += '''
           __kernel void test_compModel_Ball(
                constant double* b,
                double d,
                global double* result){
                    int gid = get_global_id(0);
                    result[gid] = cmBall(b[gid], d);
                }
        '''

        b = np.array([[0],
                   [574888093.166964],
                   [1027930772.90123],
                   [1951349330.26761],
                   [3315648910.87867]]).astype(np.float64)
        result = np.zeros((b.shape[0], 1)).astype(np.float64)
        b_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        global_range = (b.shape[0], )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel_Ball(self.queue, global_range, local_range, b_buf, np.float64(1e-9), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)

        expected = np.array([[1],
                          [0.562767842850305],
                          [0.357746453872341],
                          [0.142082226335835],
                          [0.0363104786661652]])

        assert_allclose(result, expected)

    def test_compModel_Stick(self):
        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'Stick.cl')), 'r').read()
        kernel_source += '''
           __kernel void test_compModel_Stick(
                constant double* scheme,
                double d,
                double theta,
                double phi,
                global double* result){
                    int gid = get_global_id(0);
                    double4 g = (double4)(scheme[gid * 4 + 0], scheme[gid * 4 + 1], scheme[gid * 4 + 2], 0);
                    result[gid] = cmStick(g, scheme[gid * 4 + 3], d, theta, phi);
                }
        '''

        scheme = np.array([[0, 0, 0, 0],
                   [0.0745568513761719, 0.250842839044085, -0.965152395227394, 574888093.166964],
                   [0.0396160000000000, 0.998790000000000, -0.0291910000000000, 1027930772.90123],
                   [-0.800040000000000, -0.00908930000000000, -0.599880000000000, 1951349330.26761],
                   [0.733000000000000, 0.322750000000000, 0.598780000000000, 3315648910.87867]]).astype(np.float64)
        result = np.zeros((scheme.shape[0], 1)).astype(np.float64)

        scheme_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scheme)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        global_range = (scheme.shape[0], )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel_Stick(self.queue, global_range, local_range, scheme_buf, np.float64(1e-9),
                                     np.float64(1/2.0*np.pi), np.float64(1/2.0*np.pi), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)

        expected = np.array([[1],
                          [0.964473248382023],
                          [0.358636949317381],
                          [0.999838801538188],
                          [0.707949110885549]])
        
        assert_allclose(result, expected)

    def test_compModel_Tensor(self):
        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'Tensor.cl')), 'r').read()
        kernel_source += '''
           __kernel void test_compModel(
                constant double* scheme,
                double d,
                double theta,
                double phi,
                double d_perp1,
                double d_perp2,
                double alpha,
                global double* result){
                    int gid = get_global_id(0);

                    double4 g = (double4)(scheme[gid * 4 + 0], scheme[gid * 4 + 1], scheme[gid * 4 + 2], 0.0);
                    double b = scheme[gid * 4 + 3];

                    result[gid] = cmTensor(g, b, d, d_perp1, d_perp2, theta, phi, alpha);
                }
        '''

        scheme = np.array([[0, 0, 0, 0],
                   [0.0745568513761719, 0.250842839044085, -0.965152395227394, 574888093.166964],
                   [0.0396160000000000, 0.998790000000000, -0.0291910000000000, 1027930772.90123],
                   [-0.800040000000000, -0.00908930000000000, -0.599880000000000, 1951349330.26761],
                   [0.733000000000000, 0.322750000000000, 0.598780000000000, 3315648910.87867]]).astype(np.float64)
        result = np.zeros((scheme.shape[0], 1)).astype(np.float64)

        scheme_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scheme)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        global_range = (scheme.shape[0], )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel(self.queue, global_range, local_range, scheme_buf, np.float64(1.0e-9),
                                     np.float64(1/2.0*np.pi), np.float64(1/2.0*np.pi), np.float64(1.7e-10),
                                     np.float64(1.7e-11), np.float64(1.5), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)

        expected = np.array([[1],
                          [0.953971960485703],
                          [0.358524266423466],
                          [0.815635542191575],
                          [0.529106172964909]])

        assert_allclose(result, expected)

    def test_compModel_Zeppelin(self):
        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'Zeppelin.cl')), 'r').read()
        kernel_source += '''
           __kernel void test_compModel(
                constant double* scheme,
                double d,
                double theta,
                double phi,
                double d_perp1,
                global double* result){
                    int gid = get_global_id(0);

                    double4 g = (double4)(scheme[gid * 4 + 0], scheme[gid * 4 + 1], scheme[gid * 4 + 2], 0.0);
                    double b = scheme[gid * 4 + 3];

                    result[gid] = cmZeppelin(g, b, d, d_perp1, theta, phi);
                }
        '''

        scheme = np.array([[0, 0, 0, 0],
                   [0.0745568513761719, 0.250842839044085, -0.965152395227394, 574888093.166964],
                   [0.0396160000000000, 0.998790000000000, -0.0291910000000000, 1027930772.90123],
                   [-0.800040000000000, -0.00908930000000000, -0.599880000000000, 1951349330.26761],
                   [0.733000000000000, 0.322750000000000, 0.598780000000000, 3315648910.87867]]).astype(np.float64)
        result = np.zeros((scheme.shape[0], 1)).astype(np.float64)

        scheme_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scheme)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        global_range = (scheme.shape[0], )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel(self.queue, global_range, local_range, scheme_buf, np.float64(1e-9),
                                     np.float64(1/2.0*np.pi), np.float64(1/2.0*np.pi), np.float64(1.7e-10), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)

        expected = np.array([[1],
                          [0.880069217848180],
                          [0.358485408861935],
                          [0.717585487823308],
                          [0.427274652207702]])

        assert_allclose(result, expected)

    def test_compModel_Noddi_EC(self):
        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'lib', 'cerf', 'im_w_of_x.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'lib', 'cerf', 'dawson.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'lib', 'cerf', 'erfi.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'Noddi_EC.cl')), 'r').read()
        kernel_source += '''
           __kernel void test_compModel(
                constant double* scheme,
                double d,
                double theta,
                double phi,
                double d_perp1,
                double kappa,
                global double* result){
                    int gid = get_global_id(0);

                    double4 g = (double4)(scheme[gid * 7 + 0], scheme[gid * 7 + 1], scheme[gid * 7 + 2], 0);
                    double G = scheme[gid * 7 + 3];
                    double Delta = scheme[gid * 7 + 4];
                    double delta = scheme[gid * 7 + 5];

                    double b = 7.160906424169000e+16 * pown(G * delta, 2) * (Delta - (delta/3.0));

                    result[gid] = cmNoddi_EC(g, b, d, d_perp1, theta, phi, kappa);
                }
        '''

        scheme = np.array([[0, 0, 0, 0, 0.016, 0.0079, 0.0550],
                   [0.0745568513761719, 0.250842839044085, -0.965152395227394, 0.0981000000000000, 0.016, 0.0079, 0.0550], # line 21 of Skyra scheme
                   [0.0396160000000000, 0.998790000000000, -0.0291910000000000, 0.046, 0.0289, 0.0171, 0.0740], # line 181
                   [-0.800040000000000, -0.00908930000000000, -0.599880000000000, 0.0502, 0.0337, 0.02, 0.0790], #line 261
                   [0.733000000000000, 0.322750000000000, 0.598780000000000, 0.0845, 0.0264, 0.0178, 0.0750]]).astype(np.float64)  #line 341
        result = np.zeros((scheme.shape[0], 1)).astype(np.float64)

        scheme_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scheme)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        global_range = (scheme.shape[0], )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel(self.queue, global_range, local_range, scheme_buf, np.float64(1.7e-9),
                np.float64(1/2.0*np.pi), np.float64(1/2.0*np.pi), np.float64(1.7e-10), np.float64(1), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)

        expected = np.array([[1],
                          [0.699975172696221],
                          [0.427728409321932],
                          [0.306113561328848],
                          [0.124007433956163]])

        assert_allclose(result, expected)

    def test_compModel_Noddi_IC(self):
        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'lib', 'MRIConstants.h')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'lib', 'NeumannCylPerpPGSESum.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes',
                                                             'opencl', 'lib', 'cerf',
                                                             'im_w_of_x.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes',
                                                             'opencl', 'lib', 'cerf', 'dawson.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes',
                                                             'opencl', 'lib', 'cerf', 'erfi.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes',
                                                             'opencl', 'lib',
                                                             'firstLegendreTerm.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                             'dmri', 'modelFunctions', 'Noddi_IC.cl')), 'r').read()
        kernel_source += '''
           __kernel void test_compModel(
                constant double* scheme,
                double d,
                double theta,
                double phi,
                double kappa,
                double R,
                global double* CLJnpZeros,
                int CLJnpZerosLength,
                global double* result){
                    int gid = get_global_id(0);

                    double4 g = (double4)(scheme[gid * 7 + 0], scheme[gid * 7 + 1], scheme[gid * 7 + 2], 0.0);
                    double G = scheme[gid * 7 + 3];
                    double Delta = scheme[gid * 7 + 4];
                    double delta = scheme[gid * 7 + 5];

                    double b = 7.160906424169000e+16 * pown(G * delta, 2) * (Delta - (delta/3.0));

                    result[gid] = cmNoddi_IC(g, b, G, Delta, delta, d, theta, phi, kappa, R, CLJnpZeros,
                                             CLJnpZerosLength);
                }
        '''

        scheme = np.array([[0, 0, 0, 0, 0.016, 0.0079, 0.0550],
                   [0.0745568513761719, 0.250842839044085, -0.965152395227394, 0.0981000000000000, 0.016, 0.0079, 0.0550], # line 21 of Skyra scheme
                   [0.0396160000000000, 0.998790000000000, -0.0291910000000000, 0.046, 0.0289, 0.0171, 0.0740], # line 181
                   [-0.800040000000000, -0.00908930000000000, -0.599880000000000, 0.0502, 0.0337, 0.02, 0.0790], #line 261
                   [0.733000000000000, 0.322750000000000, 0.598780000000000, 0.0845, 0.0264, 0.0178, 0.0750]]).astype(np.float64)  #line 341

        bessels = get_bessel_roots(number_of_roots=60)
        bessel_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bessels)

        result = np.zeros((scheme.shape[0], 1)).astype(np.float64)

        scheme_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scheme)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        global_range = (scheme.shape[0], )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel(self.queue, global_range, local_range, scheme_buf, np.float64(1.7e-9),
                np.float64(1/2.0*np.pi), np.float64(1/2.0*np.pi), np.float64(1.0), np.float64(1e-5),
                bessel_buf, np.int32(60), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)

        expected = np.array([[1],
                          [0.534662682033597],
                          [0.360981972106780],
                          [0.230077957975287],
                          [0.068530755182145]])
        assert_allclose(result, expected)

    def test_compModel_CylinderGPD(self):
        kernel_source = open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'lib', 'MRIConstants.h')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'lib', 'NeumannCylPerpPGSESum.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'CylinderGPD.cl')), 'r').read()
        kernel_source += '''
           __kernel void test_compModel(
                constant double* scheme,
                double d,
                double theta,
                double phi,
                double R,
                global double* CLJnpZeros,
                int CLJnpZerosLength,
                global double* result){
                    int gid = get_global_id(0);

                    double4 g = (double4)(scheme[gid * 7 + 0], scheme[gid * 7 + 1], scheme[gid * 7 + 2], 0.0);
                    double G = scheme[gid * 7 + 3];
                    double Delta = scheme[gid * 7 + 4];
                    double delta = scheme[gid * 7 + 5];

                    result[gid] = cmCylinderGPD(g, G, Delta, delta, d, theta, phi, R, CLJnpZeros, CLJnpZerosLength);
                }
        '''

        scheme = np.array([[0, 0, 0, 0, 0.016, 0.0079, 0.0550],
                   [0.0745568513761719, 0.250842839044085, -0.965152395227394, 0.0981000000000000, 0.016, 0.0079, 0.0550], # line 21 of Skyra scheme
                   [0.0396160000000000, 0.998790000000000, -0.0291910000000000, 0.046, 0.0289, 0.0171, 0.0740], # line 181
                   [-0.800040000000000, -0.00908930000000000, -0.599880000000000, 0.0502, 0.0337, 0.02, 0.0790], #line 261
                   [0.733000000000000, 0.322750000000000, 0.598780000000000, 0.0845, 0.0264, 0.0178, 0.0750]]).astype(np.float64)  #line 341

        bessels = get_bessel_roots(number_of_roots=60)
        bessel_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bessels)

        result = np.zeros((scheme.shape[0], 1)).astype(np.float64)

        scheme_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scheme)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        global_range = (scheme.shape[0], )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel(self.queue, global_range, local_range, scheme_buf, np.float64(1.7e-9),
                np.float64(1/2.0*np.pi), np.float64(1/2.0*np.pi), np.float64(2.0e-6), bessel_buf,
                np.int32(60), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)

        expected = np.array([[1],
                          [0.934370560069115],
                          [0.174948396094053],
                          [0.994957044144646],
                          [0.549962062979880]])

        assert_allclose(result, expected)

    def test_compModel_Noddi(self):
        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'lib', 'MRIConstants.h')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'lib', 'NeumannCylPerpPGSESum.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes',
                                                             'opencl', 'lib', 'cerf',
                                                             'im_w_of_x.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes',
                                                             'opencl', 'lib', 'cerf', 'dawson.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes',
                                                             'opencl', 'lib', 'cerf', 'erfi.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.  join(os.path.dirname(__file__), '..', '..', 'includes',
                                                             'opencl',
                                                             'lib', 'firstLegendreTerm.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                             'dmri',  'modelFunctions', 'Ball.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                             'dmri',  'modelFunctions', 'Noddi_EC.cl')), 'r').read() + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'Noddi_IC.cl')), 'r').read() + "\n"

        kernel_source += '''
           __kernel void test_compModel(
                constant double* scheme,
                double w_ic,
                double w_csf,
                double theta,
                double phi,
                double kappa,
                double R,
                global double* CLJnpZeros,
                int CLJnpZerosLength,
                global double* result){
                    int gid = get_global_id(0);

                    double4 g = (double4)(scheme[gid * 6 + 0], scheme[gid * 6 + 1], scheme[gid * 6 + 2], 0.0);
                    double G = scheme[gid * 6 + 3];
                    double Delta = scheme[gid * 6 + 4];
                    double delta = scheme[gid * 6 + 5];

                    double b = 7.160906424169000e+16 * pown(G * delta, 2) * (Delta - (delta/3.0));

                    result[gid] = (
                                        (w_csf * cmBall(b, 3.0e-9)) +
                                        (1-w_csf) * (
                                            ((1-w_ic) * cmNoddi_EC(g, b, 1.7e-9, 1.7e-9*(1 - w_ic), theta, phi,
                                                                    kappa)) +
                                            (w_ic * cmNoddi_IC(g, b, G, Delta, delta, 1.7e-9, theta, phi, kappa, R,
                                                                CLJnpZeros, CLJnpZerosLength))
                                        )
                                  );
                }
        '''

        dataset = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                      '..', '..', 'includes', 'test_data',
                                                                      'noddi', 'noddi_example.mat')))

        bessels = get_bessel_roots(number_of_roots=60)
        bessel_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bessels)

        scheme = dataset["scheme"].astype(np.float64)
        result = np.zeros((scheme.shape[0], 1)).astype(np.float64)
        global_range = (scheme.shape[0], )
        scheme = scheme.reshape((-1))
        scheme_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scheme)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel(self.queue, global_range, local_range, scheme_buf, np.float64(1/3.0), np.float64(1/3.0),
                               np.float64(0.422355757924254), np.float64(0.765185229784043), np.float64(0.5),
                               np.float64(0.0), bessel_buf, np.int32(60), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)
        expected = dataset['result']
        assert_allclose(result, expected)

    def test_compModel_T2(self):
        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'includes', 'opencl',
                                              'dmri', 'modelFunctions', 'ExpT2Dec.cl')), 'r').read()
        kernel_source += '''
           __kernel void test_compModel_T2(
                constant double* TE,
                double T2_list,
                global double* result){
                    int gid = get_global_id(0);
                    result[gid] = cmExpT2Dec(TE[gid], T2_list);
                }
        '''

        TE = np.array([[0.04], [0.045], [0.05], [0.06], [0.07], [0.08],
                       [0.09], [0.1], [0.11], [0.12], [0.13], [0.14]]).astype(np.float64)
        result = np.zeros((TE.shape[0], 1)).astype(np.float64)
        TE_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=TE)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        global_range = (TE.shape[0], )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.test_compModel_T2(self.queue, global_range, local_range, TE_buf, np.float64(0.05), result_buf)
        cl.enqueue_copy(self.queue, result, result_buf)

        expected = np.array([[0.449328964117222],
                                [0.406569659740599],
                                [0.367879441171442],
                                [0.301194211912202],
                                [0.246596963941606],
                                [0.201896517994655],
                                [0.165298888221587],
                                [0.135335283236613],
                                [0.110803158362334],
                                [0.0907179532894125],
                                [0.0742735782143339],
                                [0.0608100626252180]])

        assert_allclose(result, expected)

    def test_lmmin(self):
        parameters = np.array([[100, 0, -10],
                               [100, 0, -10]])
        var_data = np.array([[16.6, 9.9, 4.4, 1.1, 0., 1.1, 4.2, 9.3, 16.4],
                            [-15, -8, -3, 0, 1, 0, -3, -8, -15]])
        prtcl_data = np.array([-4., -3., -2., -1.,  0., 1.,  2.,  3.,  4.])

        nmr_params = parameters.shape[1]
        nmr_inst_per_problem = var_data.shape[1]
        nmr_problems = var_data.shape[0]

        prtcl_data = np.reshape(prtcl_data, (-1, 1)).astype(np.float64)
        var_data = np.reshape(var_data, (-1, 1)).astype(np.float64)
        parameters = np.reshape(parameters, (-1, 1)).astype(np.float64)

        kernel_source = get_cl_double_extension_definer(self.platform) + "\n"
        kernel_source += '''
            #define NMR_PARAMS ''' + repr(nmr_params) + '''
            #define NMR_INST_PER_PROBLEM ''' + repr(nmr_inst_per_problem) + '''
        '''

        kernel_source += '''
            typedef struct{
                global double* problemSet;
                global double* prtcl_data_1;
            } optimize_data;
        '''

        kernel_source += '''
            void evaluate(
                const void* data,
                double* local_params,
                double* result){
                    int gid = get_global_id(0);

                    optimize_data* D;
                    D = (optimize_data*)data;

                    for(int i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = D->problemSet[gid * NMR_INST_PER_PROBLEM + i] - (local_params[0] + local_params[1]
                                        * D->prtcl_data_1[i] + local_params[2] * D->prtcl_data_1[i]
                                        * D->prtcl_data_1[i]);
                    }
                }
        '''
        kernel_source += LMMin(nmr_params).get_cl_code()
        kernel_source += '''
            __kernel void minimize(
                global double* prtcl_data_1,
                global double* problemSet,
                global double* params){
                    int gid = get_global_id(0);

                    double x[NMR_PARAMS];
                    for(int i = 0; i < NMR_PARAMS; i++){
                        x[i] = params[gid * NMR_PARAMS + i];
                    }

                    optimize_data data = {problemSet, prtcl_data_1};

                    lmmin(x, (const void*) &data);

                    for(int i = 0; i < NMR_PARAMS; i++){
                        params[gid * NMR_PARAMS + i] = x[i];
                    }
            }
        '''

        prtcl_data_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=prtcl_data)
        var_data_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=var_data)
        parameters_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=parameters)

        global_range = (nmr_problems, )
        local_range = None
        program = cl.Program(self.context, kernel_source).build()
        program.minimize(self.queue, global_range, local_range, prtcl_data_buf, var_data_buf, parameters_buf)
        cl.enqueue_copy(self.queue, parameters, parameters_buf)

        parameters = np.reshape(parameters, (nmr_problems, nmr_params))

        expected = np.reshape(np.array([[0.12987013, -0.05, 1.03051948],
                                        [1, -2.57432389e-18, -1]]), (nmr_problems, nmr_params))
        assert_allclose(parameters, expected, rtol=1e-5)
