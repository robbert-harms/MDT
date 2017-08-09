import numpy as np

from mdt.utils import eigen_vectors_from_tensor
from mot.cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2015-04-16"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DTIMeasures(CLRoutine):

    def calculate(self, parameters_dict):
        """Calculate DTI statistics from the given eigenvalues.

        Args:
            parameters_dict (dict): Dictionary containing at least theta, phi, psi, d, dperp0 and dperp1
                We will use this to generate some standard measures from the diffusion Tensor.

        Returns:
            dict: as keys typical elements like 'FA, 'MD', 'eigval' etc. and as per values the maps.
                These maps are per voxel, and optionally per instance per voxel
        """
        eigenvectors = np.stack(eigen_vectors_from_tensor(np.squeeze(parameters_dict['theta']),
                                                          np.squeeze(parameters_dict['phi']),
                                                          np.squeeze(parameters_dict['psi'])), axis=0)

        eigenvalues = np.atleast_2d(np.squeeze(np.dstack([parameters_dict['d'],
                                                          parameters_dict['dperp0'],
                                                          parameters_dict['dperp1']])))

        sorted_eigenvalues, sorted_eigenvectors, ranking = self._sort_eigensystem(eigenvalues, eigenvectors)

        md = np.sum(sorted_eigenvalues, axis=1) / 3.
        fa = np.sqrt(3/2.) * (np.sqrt(np.sum((sorted_eigenvalues - md[..., None]) ** 2, axis=1))
                              / np.sqrt(np.sum(sorted_eigenvalues**2, axis=1)))

        output = {'FA': fa,
                  'MD': md,
                  'AD': sorted_eigenvalues[:, 0],
                  'RD': (sorted_eigenvalues[:, 1] + sorted_eigenvalues[:, 2]) / 2.0,
                  'eigen_ranking': ranking,
                  'sorted_d': sorted_eigenvalues[:, 0],
                  'sorted_dperp0': sorted_eigenvalues[:, 1],
                  'sorted_dperp1': sorted_eigenvalues[:, 2]
                  }

        for ind in range(3):
            output.update({'sorted_vec{}'.format(ind): sorted_eigenvectors[ind]})

        return output

    def get_output_names(self):
        """Get a list of the map names calculated by this class.

        Returns:
            list of str: the list of map names this calculator returns
        """
        return_names = ['FA', 'MD', 'AD', 'RD', 'eigen_ranking', 'sorted_d', 'sorted_dperp0', 'sorted_dperp1']
        for ind in range(3):
            return_names.append('sorted_vec{}'.format(ind))
        return return_names

    def _sort_eigensystem(self, eigenvalues, eigenvectors):
        ranking = np.atleast_2d(np.squeeze(np.argsort(eigenvalues, axis=1)[:, ::-1]))
        voxels_range = np.arange(ranking.shape[0])
        sorted_eigenvalues = np.concatenate([eigenvalues[voxels_range, ranking[:, ind], None]
                                             for ind in range(ranking.shape[1])], axis=1)
        sorted_eigenvectors = np.stack([eigenvectors[ranking[:, ind], voxels_range, :]
                                        for ind in range(ranking.shape[1])])

        return sorted_eigenvalues, sorted_eigenvectors, ranking
