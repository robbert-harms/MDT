import numpy as np
from mdt.utils import tensor_spherical_to_cartesian, tensor_cartesian_to_spherical
from mot.cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2015-04-16"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DTIMeasures(CLRoutine):

    def calculate(self, parameters_dict):
        """Post process the DTI results.

        This will re-orient the Tensor such that the eigen values are in decreasing order. This is done by
        permuting the eigen values and eigen vectors and then create theta, phi and psi to match the rotated system.

        After that, some interesting measures like FA, MD, RD and AD are added to the results.

        Args:
            parameters_dict (dict): Dictionary containing at least theta, phi, psi, d, dperp0 and dperp1
                We will use this to generate some standard measures from the diffusion Tensor.

        Returns:
            dict: as keys typical elements like 'FA, 'MD', 'eigval' etc. and as per values the maps.
                These maps are per voxel, and optionally per instance per voxel
        """
        sorted_eigenvalues, sorted_eigenvectors, ranking = self._sort_eigensystem(parameters_dict)
        theta, phi, psi = tensor_cartesian_to_spherical(sorted_eigenvectors[0], sorted_eigenvectors[1])

        md = np.sum(sorted_eigenvalues, axis=1) / 3.
        fa = np.sqrt(3/2.) * (np.sqrt(np.sum((sorted_eigenvalues - md[..., None]) ** 2, axis=1))
                              / np.sqrt(np.sum(sorted_eigenvalues**2, axis=1)))

        output = {'FA': fa,
                  'MD': md,
                  'AD': sorted_eigenvalues[:, 0],
                  'RD': (sorted_eigenvalues[:, 1] + sorted_eigenvalues[:, 2]) / 2.0,
                  'd': sorted_eigenvalues[:, 0],
                  'dperp0': sorted_eigenvalues[:, 1],
                  'dperp1': sorted_eigenvalues[:, 2],
                  'theta': theta,
                  'phi': phi,
                  'psi': psi
                  }

        for ind in range(3):
            output.update({'vec{}'.format(ind): sorted_eigenvectors[ind]})

        return output

    def get_output_names(self):
        """Get a list of the map names calculated by this class.

        Returns:
            list of str: the list of map names this calculator returns
        """
        return_names = ['FA', 'MD', 'AD', 'RD', 'd', 'dperp0', 'dperp1', 'theta', 'phi', 'psi']
        for ind in range(3):
            return_names.append('vec{}'.format(ind))
        return return_names

    def _sort_eigensystem(self, parameters_dict):
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
