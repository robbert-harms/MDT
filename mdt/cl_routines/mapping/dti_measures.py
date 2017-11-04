import numpy as np
from mdt.utils import tensor_spherical_to_cartesian, tensor_cartesian_to_spherical
from mot.cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2015-04-16"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DTIMeasures(CLRoutine):

    @staticmethod
    def post_optimization_modifier(parameters_dict):
        """Apply post optimization modification of the Tensor compartment.

        This will re-orient the Tensor such that the eigen values are in decreasing order. This is done by
        permuting the eigen values and eigen vectors and then recreating theta, phi and psi to match the rotated system.

        Args:
            parameters_dict (dict): the results from optimization

        Returns:
            dict: same set of parameters but then possibly updated with a rotation.
        """
        sorted_eigenvalues, sorted_eigenvectors, ranking = DTIMeasures.sort_eigensystem(parameters_dict)
        theta, phi, psi = tensor_cartesian_to_spherical(sorted_eigenvectors[0], sorted_eigenvectors[1])
        return {'d': sorted_eigenvalues[:, 0], 'dperp0': sorted_eigenvalues[:, 1], 'dperp1': sorted_eigenvalues[:, 2],
                'theta': theta, 'phi': phi, 'psi': psi}

    @staticmethod
    def sort_eigensystem(parameters_dict):
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

    def calculate(self, results):
        """Return some interesting measures like FA, MD, RD and AD.

        Args:
            results (dict): Dictionary containing at least theta, phi, psi, d, dperp0 and dperp1
                We will use this to generate some standard measures from the diffusion Tensor.

        Returns:
            dict: as keys typical elements like 'FA and 'MD' as interesting output and as per values the maps.
                These maps are per voxel, and optionally per instance per voxel
        """
        md = (results['d'] + results['dperp0'] + results['dperp1']) / 3.
        md_std = np.sqrt(results['d.std'] + results['dperp0.std'] + results['dperp1.std']) / 3.

        fa = DTIMeasures.fractional_anisotropy(results['d'], results['dperp0'], results['dperp1'])
        fa_std = DTIMeasures.fractional_anisotropy_std(
            results['d'], results['dperp0'], results['dperp1'],
            results['d.std'], results['dperp0.std'], results['dperp1.std'])

        output = {
            'FA': fa,
            'FA.std': fa_std,
            'MD': md,
            'MD.std': md_std,
            'AD': results['d'],
            'AD.std': results['d.std'],
            'RD': (results['dperp0'] + results['dperp1']) / 2.0,
            'RD.std': (results['dperp0.std'] + results['dperp1.std']) / 2.0,
        }

        if all(el in results for el in ['theta', 'phi', 'psi']):
            eigenvectors = tensor_spherical_to_cartesian(np.squeeze(results['theta']),
                                                         np.squeeze(results['phi']),
                                                         np.squeeze(results['psi']))
            for ind in range(3):
                output.update({'vec{}'.format(ind): eigenvectors[ind]})

        return output

    @staticmethod
    def fractional_anisotropy(d, dperp0, dperp1):
        """Calculate the fractional anisotropy (FA).

        Returns:
            ndarray: the fractional anisotropy for each voxel.
        """
        d, dperp0, dperp1 = map(lambda el: np.squeeze(el * 1e10), [d, dperp0, dperp1])
        return np.sqrt(1/2.) * np.sqrt(((d - dperp0)**2 + (dperp0 - dperp1)**2 + (dperp1 - d)**2)
                                       / (d**2 + dperp0**2 + dperp1**2))

    @staticmethod
    def fractional_anisotropy_std(d, dperp0, dperp1, d_std, dperp0_std, dperp1_std):
        """Calculate the standard deviation of the fractional anisotropy (FA) using error propagation.

        Returns:
            the standard deviation of the fraction anisotropy using error propagation of the diffusivities.
        """
        d, dperp0, dperp1, d_std, dperp0_std, dperp1_std = \
            map(lambda el: np.squeeze(el * 1e10), [d, dperp0, dperp1, d_std, dperp0_std, dperp1_std])

        return 1/2. * np.sqrt((d + dperp0 + dperp1) ** 2 * (
            d_std ** 2 * (-d * (dperp0 + dperp1) + dperp0 ** 2 + dperp1 ** 2) ** 2 +
            dperp0_std ** 2 * (d ** 2 - d * dperp0 + dperp1 * (dperp1 - dperp0)) ** 2 +
            dperp1_std ** 2 * (d ** 2 - d * dperp1 + dperp0 * (dperp0 - dperp1)) ** 2
        ) / ((d ** 2 + dperp0 ** 2 + dperp1 ** 2) ** 3 *
             (d ** 2 - d * (dperp0 + dperp1) + dperp0 ** 2 - dperp0 * dperp1 + dperp1 ** 2)))
