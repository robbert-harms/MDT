import numpy as np
import pyopencl as cl
from mot.utils import get_float_type_def
from mot.cl_routines.base import AbstractCLRoutine
from mot.load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2015-04-16"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DTIMeasures(AbstractCLRoutine):

    def concat_and_calculate(self, eigenval1, eigenval2, eigenval3, double_precision=True):
        """Calculate DTI statistics from the given eigenvalues.

        This concatenates the eigenvalue matrices and runs self.calculate(eigenvalues) on them.

        Args:
            eigenval1 (ndarray):
                The first set of eigenvalues, can be 2d, per voxel one eigenvalue,
                or 3d, per voxel multiple eigenvalues.
            eigenval2 (ndarray):
                The first set of eigenvalues, can be 2d, per voxel one eigenvalue,
                or 3d, per voxel multiple eigenvalues.
            eigenval3 (ndarray):
                The first set of eigenvalues, can be 2d, per voxel one eigenvalue,
                or 3d, per voxel multiple eigenvalues.
            double_precision (boolean): if we want to use float (set it to False) or double (set it to True)

        Returns:
            Per voxel, and optionally per instance per voxel, the FA and MD: (fa, md)
        """
        s = eigenval1.shape
        if len(s) < 3:
            return self.calculate(np.concatenate((np.reshape(eigenval1, (s[0], 1)),
                                                  np.reshape(eigenval2, (s[0], 1)),
                                                  np.reshape(eigenval3, (s[0], 1))), axis=1), double_precision)
        else:
            return self.calculate(np.concatenate((np.reshape(eigenval1, (s[0], s[1], 1)),
                                                  np.reshape(eigenval2, (s[0], s[1], 1)),
                                                  np.reshape(eigenval3, (s[0], s[1], 1))), axis=2), double_precision)

    def calculate(self, eigenvalues, double_precision=True):
        """Calculate DTI statistics from the given eigenvalues.

        Args:
            eigenvalues (ndarray):
                The set of eigen values, can be 2d, per voxel one eigenvalue, or 3d, per voxel multiple eigenvalues.
            double_precision (boolean): if we want to use float (set it to False) or double (set it to True)

        Returns:
            Per voxel, and optionally per instance per voxel, the FA and MD: (fa, md)
        """
        np_dtype = np.float32
        if double_precision:
            np_dtype = np.float64

        eigenvalues = eigenvalues.astype(np_dtype, order='C', copy=False)

        s = eigenvalues.shape
        if len(s) < 3:
            fa_host = np.zeros((s[0], 1), dtype=np_dtype)
            md_host = np.zeros((s[0], 1), dtype=np_dtype)
            items = s[0]
        else:
            fa_host = np.zeros((s[0] * s[1], 1), dtype=np_dtype)
            md_host = np.zeros((s[0] * s[1], 1), dtype=np_dtype)
            items = s[0] * s[1]
            eigenvalues = np.reshape(eigenvalues, (s[0] * s[1], -1))

        workers = self._create_workers(_DTIMeasuresWorker, [eigenvalues, fa_host, md_host, double_precision])
        self.load_balancer.process(workers, items)

        if len(s) > 2:
            fa_host = np.reshape(fa_host, (s[0], s[1], 1))
            md_host = np.reshape(md_host, (s[0], s[1], 1))
        return fa_host, md_host


class _DTIMeasuresWorker(Worker):

    def __init__(self, cl_environment, eigenvalues, fa_host, md_host, double_precision):
        super(_DTIMeasuresWorker, self).__init__(cl_environment)
        self._eigenvalues = eigenvalues
        self._fa_host = fa_host
        self._md_host = md_host
        self._double_precision = double_precision
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        write_flags = self._cl_environment.get_write_only_cl_mem_flags()
        read_flags = self._cl_environment.get_read_only_cl_mem_flags()

        eigenvalues_buf = cl.Buffer(self._cl_run_context.context, read_flags,
                                    hostbuf=self._eigenvalues[range_start:range_end])
        fa_buf = cl.Buffer(self._cl_run_context.context, write_flags, hostbuf=self._fa_host[range_start:range_end])
        md_buf = cl.Buffer(self._cl_run_context.context, write_flags, hostbuf=self._md_host[range_start:range_end])
        buffers = [eigenvalues_buf, fa_buf, md_buf]

        self._kernel.calculate_measures(self._cl_run_context.queue, (int(nmr_problems), ), None, *buffers)
        cl.enqueue_copy(self._cl_run_context.queue, self._fa_host[range_start:range_end], fa_buf, is_blocking=True)
        event = cl.enqueue_copy(self._cl_run_context.queue, self._md_host[range_start:range_end], md_buf, is_blocking=False)

        return event

    def _get_kernel_source(self):
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += '''
            __kernel void calculate_measures(
                global MOT_FLOAT_TYPE* eigenvalues,
                global MOT_FLOAT_TYPE* fas,
                global MOT_FLOAT_TYPE* mds
                ){
                    int gid = get_global_id(0);
                    int voxel = gid * 3;

                    MOT_FLOAT_TYPE v1 = eigenvalues[voxel];
                    MOT_FLOAT_TYPE v2 = eigenvalues[voxel + 1];
                    MOT_FLOAT_TYPE v3 = eigenvalues[voxel + 2];

                    fas[gid] = sqrt(0.5 * (pown(v1 - v2, 2) + pown(v1 - v3, 2) + pown(v2 - v3, 2)) /
                                           (pown(v1, 2) + pown(v2, 2) + pown(v3, 2)));
                    mds[gid] = (v1 + v2 + v3) / 3.0;
            }
        '''
        return kernel_source