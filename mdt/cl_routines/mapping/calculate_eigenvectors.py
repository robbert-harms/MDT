import pyopencl as cl
import numpy as np
from mot.utils import get_float_type_def
from mot.cl_routines.base import CLRoutine
from mot.load_balance_strategies import Worker
from mdt.components_loader import LibraryFunctionsLoader


__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CalculateEigenvectors(CLRoutine):

    def convert_theta_phi_psi(self, theta_roi, phi_roi, psi_roi, double_precision=False):
        """Calculate the eigenvectors from the given theta, phi and psi angles.

        This will return the eigenvectors unsorted (since we know nothing about the eigenvalues).

        Args:
            theta_roi (ndarray): The list of theta's per voxel in the ROI
            phi_roi (ndarray): The list of phi's per voxel in the ROI
            psi_roi (ndarray): The list of psi's per voxel in the ROI
            double_precision (boolean): if we want to use float (set it to False) or double (set it to True)

        Returns:
            The three eigenvectors per voxel in the ROI. The return matrix is of shape (n, 3, 3) where n is the number
            of voxels and the second dimension holds the number of vectors and the last dimension the direction
            per vector. In other words, this gives for one voxel the matrix::

                [evec_1_x, evec_1_y, evec_1_z,
                 evec_2_x, evec_2_y, evec_2_z
                 evec_3_x, evec_3_y, evec_3_z]

            The resulting eigenvectors are the same as those from the Tensor compartment model.
        """
        np_dtype = np.float32
        if double_precision:
            np_dtype = np.float64

        theta_roi = np.require(theta_roi, np_dtype, requirements=['C', 'A', 'O'])
        phi_roi = np.require(phi_roi, np_dtype, requirements=['C', 'A', 'O'])
        psi_roi = np.require(psi_roi, np_dtype, requirements=['C', 'A', 'O'])

        rows = theta_roi.shape[0]
        evecs = np.zeros((rows, 3, 3), dtype=np_dtype, order='C')

        workers = self._create_workers(lambda cl_environment: _CEWorker(cl_environment,
                                                                        self.get_compile_flags_list(double_precision),
                                                                        theta_roi, phi_roi,
                                                                        psi_roi, evecs, double_precision))
        self.load_balancer.process(workers, rows)
        return evecs


class _CEWorker(Worker):

    def __init__(self, cl_environment, compile_flags, theta_roi, phi_roi, psi_roi, evecs, double_precision):
        super(_CEWorker, self).__init__(cl_environment)

        self._theta_roi = theta_roi
        self._phi_roi = phi_roi
        self._psi_roi = psi_roi
        self._evecs = evecs
        self._double_precision = double_precision
        self._all_buffers, self._evecs_buf = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def __del__(self):
        list(buffer.release() for buffer in self._all_buffers)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        self._kernel.generate_tensor(self._cl_run_context.queue, (int(nmr_problems), ), None, *self._all_buffers,
                                     global_offset=(int(range_start),))
        self._enqueue_readout(self._evecs_buf, self._evecs, range_start, range_end)

    def _create_buffers(self):
        thetas_buf = cl.Buffer(self._cl_run_context.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self._theta_roi)
        phis_buf = cl.Buffer(self._cl_run_context.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self._phi_roi)
        psis_buf = cl.Buffer(self._cl_run_context.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self._psi_roi)
        evecs_buf = cl.Buffer(self._cl_run_context.context,
                              cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                              hostbuf=self._evecs)

        all_buffers = [thetas_buf, phis_buf, psis_buf, evecs_buf]
        return all_buffers, evecs_buf

    def _get_kernel_source(self):
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)

        lib_loader = LibraryFunctionsLoader()
        kernel_source += lib_loader.load('TensorSphericalToCartesian').get_cl_code()

        kernel_source += '''
            __kernel void generate_tensor(
                global mot_float_type* thetas,
                global mot_float_type* phis,
                global mot_float_type* psis,
                global mot_float_type* evecs
                ){
                    ulong gid = get_global_id(0);

                    mot_float_type4 vec0, vec1, vec2;
                    TensorSphericalToCartesian(thetas[gid], phis[gid], psis[gid], &vec0, &vec1, &vec2);

                    evecs[gid*9] = vec0.x;
                    evecs[gid*9 + 1] = vec0.y;
                    evecs[gid*9 + 2] = vec0.z;

                    evecs[gid*9 + 3] = vec1.x;
                    evecs[gid*9 + 4] = vec1.y;
                    evecs[gid*9 + 5] = vec1.z;

                    evecs[gid*9 + 6] = vec2.x;
                    evecs[gid*9 + 7] = vec2.y;
                    evecs[gid*9 + 8] = vec2.z;
            }
        '''
        return kernel_source
