import numpy as np
import pyopencl as cl
from mot.utils import get_float_type_def
from mot.cl_routines.base import CLRoutine
from mot.load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2015-04-16"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DTIMeasures(CLRoutine):

    def calculate(self, eigenvalues, eigenvectors):
        """Calculate DTI statistics from the given eigenvalues.

        Args:
            eigenvalues (ndarray): List of eigenvalues per voxel. This requires as 2d matrix with for every voxel
                three eigenvalues. They need not be sorted, we will sort them and the sorted maps are part
                of the output.
            eigenvectors (ndarray): List of eigenvectors per voxel. This requires a 3d matrix of the same length (in the
                first dimension) as the eigenvalues. The other two dimensions should represent a 3x3 matrix with the
                three eigenvectors.

        Returns:
            dict: as keys typical elements like 'FA, 'MD', 'eigval' etc. and as per values the maps.
                These maps are per voxel, and optionally per instance per voxel
        """
        fa_host, md_host = self._get_fa_md(eigenvalues)
        sorted_eigenvalues, sorted_eigenvectors, ranking = self._sort_eigensystem(eigenvalues, eigenvectors)

        output = {'FA': fa_host,
                  'MD': md_host,
                  'AD': sorted_eigenvalues[:, 0],
                  'RD': (sorted_eigenvalues[:, 1] + sorted_eigenvalues[:, 2]) / 2.0,
                  'eigen_ranking': ranking}

        for ind in range(3):
            output.update({'sorted_vec{}'.format(ind): sorted_eigenvectors[:, ind, :],
                           'sorted_eigval{}'.format(ind): sorted_eigenvalues[:, ind]})

        return output

    def get_output_names(self):
        """Get a list of the map names calculated by this class.

        Returns:
            list of str: the list of map names this calculator returns
        """
        return_names = ['FA', 'MD', 'AD', 'RD', 'eigen_ranking']
        for ind in range(3):
            return_names.append('sorted_vec{}'.format(ind))
            return_names.append('sorted_eigval{}'.format(ind))
        return return_names

    def _sort_eigensystem(self, eigenvalues, eigenvectors):
        ranking = np.atleast_2d(np.squeeze(np.argsort(eigenvalues, axis=1)[:, ::-1]))
        voxels_range = np.arange(ranking.shape[0])
        sorted_eigenvalues = np.concatenate([eigenvalues[voxels_range, ranking[:, ind], None]
                                             for ind in range(ranking.shape[1])], axis=1)
        sorted_eigenvectors = np.concatenate([eigenvectors[voxels_range, ranking[:, ind], None, :]
                                              for ind in range(ranking.shape[1])], axis=1)

        return sorted_eigenvalues, sorted_eigenvectors, ranking

    def _get_fa_md(self, eigenvalues):
        if eigenvalues.dtype == np.float32:
            np_dtype = np.float32
            double_precision = False
        else:
            np_dtype = np.float64
            double_precision = True

        eigenvalues = np.require(eigenvalues, np_dtype, requirements=['C', 'A', 'O'])

        s = eigenvalues.shape
        fa_host = np.zeros((s[0], 1), dtype=np_dtype)
        md_host = np.zeros((s[0], 1), dtype=np_dtype)
        nmr_voxels = s[0]

        workers = self._create_workers(lambda cl_environment: _DTIMeasuresWorker(
            cl_environment, self.get_compile_flags_list(double_precision=True), eigenvalues,
            fa_host, md_host, double_precision))
        self.load_balancer.process(workers, nmr_voxels)

        return fa_host, md_host


class _DTIMeasuresWorker(Worker):

    def __init__(self, cl_environment, compile_flags, eigenvalues, fa_host, md_host, double_precision):
        super(_DTIMeasuresWorker, self).__init__(cl_environment)
        self._eigenvalues = eigenvalues
        self._fa_host = fa_host
        self._md_host = md_host
        self._double_precision = double_precision
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        eigenvalues_buf = cl.Buffer(self._cl_run_context.context,
                                    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self._eigenvalues[range_start:range_end])
        fa_buf = cl.Buffer(self._cl_run_context.context,
                           cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self._fa_host[range_start:range_end])
        md_buf = cl.Buffer(self._cl_run_context.context,
                           cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self._md_host[range_start:range_end])

        buffers = [eigenvalues_buf, fa_buf, md_buf]

        self._kernel.calculate_measures(self._cl_run_context.queue, (int(nmr_problems), ), None, *buffers)
        cl.enqueue_copy(self._cl_run_context.queue, self._fa_host[range_start:range_end], fa_buf, is_blocking=False)
        cl.enqueue_copy(self._cl_run_context.queue, self._md_host[range_start:range_end], md_buf, is_blocking=False)

    def _get_kernel_source(self):
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += '''
            __kernel void calculate_measures(
                global mot_float_type* eigenvalues,
                global mot_float_type* fas,
                global mot_float_type* mds
                ){
                    ulong gid = get_global_id(0);
                    ulong voxel = gid * 3;

                    mot_float_type v1 = eigenvalues[voxel];
                    mot_float_type v2 = eigenvalues[voxel + 1];
                    mot_float_type v3 = eigenvalues[voxel + 2];

                    fas[gid] = sqrt(0.5 * (((v1 - v2) * (v1 - v2)) +
                                           ((v1 - v3) * (v1 - v3)) +
                                           ((v2 - v3)  * (v2 - v3))) /
                                                (v1 * v1 + v2 * v2 + v3 * v3));
                    mds[gid] = (v1 + v2 + v3) / 3.0;
            }
        '''
        return kernel_source
