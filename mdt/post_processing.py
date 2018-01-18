"""This module contains various standard post-processing routines for use after optimization or sampling."""
import numpy as np
from mdt.utils import tensor_spherical_to_cartesian, tensor_cartesian_to_spherical
from mot.utils import split_in_batches

__author__ = 'Robbert Harms'
__date__ = '2017-12-10'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class DTIMeasures(object):

    @staticmethod
    def extra_optimization_maps(results):
        """Return some interesting measures like FA, MD, RD and AD.

        Args:
            results (dict): Dictionary containing at least theta, phi, psi, d, dperp0 and dperp1
                We will use this to generate some standard measures from the diffusion Tensor.

        Returns:
            dict: as keys typical elements like 'FA and 'MD' as interesting output and as per values the maps.
                These maps are per voxel, and optionally per instance per voxel
        """
        output = {
            'FA': DTIMeasures.fractional_anisotropy(results['d'], results['dperp0'], results['dperp1']),
            'MD': (results['d'] + results['dperp0'] + results['dperp1']) / 3.,
            'AD': results['d'],
            'RD': (results['dperp0'] + results['dperp1']) / 2.0,
        }

        if all('{}.std'.format(el) in results for el in ['d', 'dperp0', 'dperp1']):
            output.update({
                'FA.std': DTIMeasures.fractional_anisotropy_std(
                    results['d'], results['dperp0'], results['dperp1'],
                    results['d.std'], results['dperp0.std'], results['dperp1.std']),
                'MD.std': np.sqrt(results['d.std'] + results['dperp0.std'] + results['dperp1.std']) / 3.,
                'AD.std': results['d.std'],
                'RD.std': (results['dperp0.std'] + results['dperp1.std']) / 2.0,
            })

        if all(el in results for el in ['theta', 'phi', 'psi']):
            eigenvectors = tensor_spherical_to_cartesian(np.squeeze(results['theta']),
                                                         np.squeeze(results['phi']),
                                                         np.squeeze(results['psi']))
            for ind in range(3):
                output.update({'vec{}'.format(ind): eigenvectors[ind]})

        return output

    @staticmethod
    def extra_sampling_maps(results):
        """Return some interesting measures derived from the samples.

        Please note that this function expects the result dictionary only with the parameter names, that is,
        it expects the elements ``d``, ``dperp0`` and ``dperp1`` to be present.

        Args:
            results (dict[str: ndarray]): a dictionary containing the samples for each of the parameters.

        Returns:
            dict: a set of additional maps with one value per voxel.
        """
        mds = (results['d'] + results['dperp0'] + results['dperp1']) / 3.
        fas = DTIMeasures.fractional_anisotropy(results['d'], results['dperp0'], results['dperp1'])
        rds = (results['dperp0'] + results['dperp1']) / 2.0

        return {
            'MD': np.mean(mds, axis=1),
            'MD.std': np.std(mds, axis=1),
            'FA': np.mean(fas, axis=1),
            'FA.std': np.std(fas, axis=1),
            'AD': np.mean(results['d'], axis=1),
            'AD.std': np.std(results['d'], axis=1),
            'RD': np.mean(rds, axis=1),
            'RD.std': np.std(rds, axis=1)
        }

    @staticmethod
    def post_optimization_modifier(parameters_dict):
        """Apply post optimization modification of the Tensor compartment.

        This will re-orient the Tensor such that the eigen values are in decreasing order. This is done by
        permuting the eigen-values and -vectors and then recreating theta, phi and psi to match the rotated system.

        This is done primarily to be able to directly use the Tensor results in MCMC sampling. Since we often put a
        prior on the diffusivities to be in decreasing order, we need to make sure that the starting point is valid.

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

    @staticmethod
    def fractional_anisotropy(d, dperp0, dperp1):
        """Calculate the fractional anisotropy (FA).

        Returns:
            ndarray: the fractional anisotropy for each voxel.
        """
        def compute(d, dperp0, dperp1):
            d, dperp0, dperp1 = map(lambda el: np.squeeze(el).astype(np.float64), [d, dperp0, dperp1])
            return np.sqrt(1 / 2.) * np.sqrt(((d - dperp0) ** 2 + (dperp0 - dperp1) ** 2 + (dperp1 - d) ** 2)
                                             / (d ** 2 + dperp0 ** 2 + dperp1 ** 2))

        if len(d.shape) > 1 and d.shape[1] > 1:
            fa = np.zeros_like(d)
            for batch_start, batch_end in split_in_batches(d.shape[1], 100):
                fa[:, batch_start:batch_end] = compute(
                    d[:, batch_start:batch_end],
                    dperp0[:, batch_start:batch_end],
                    dperp1[:, batch_start:batch_end])
            return fa
        else:
            return compute(d, dperp0, dperp1)



    @staticmethod
    def fractional_anisotropy_std(d, dperp0, dperp1, d_std, dperp0_std, dperp1_std):
        """Calculate the standard deviation of the fractional anisotropy (FA) using error propagation.

        Returns:
            the standard deviation of the fraction anisotropy using error propagation of the diffusivities.
        """
        d, dperp0, dperp1, d_std, dperp0_std, dperp1_std = \
            map(lambda el: np.squeeze(el).astype(np.float64), [d, dperp0, dperp1, d_std, dperp0_std, dperp1_std])

        return 1/2. * np.sqrt((d + dperp0 + dperp1) ** 2 * (
            d_std ** 2 * (-d * (dperp0 + dperp1) + dperp0 ** 2 + dperp1 ** 2) ** 2 +
            dperp0_std ** 2 * (d ** 2 - d * dperp0 + dperp1 * (dperp1 - dperp0)) ** 2 +
            dperp1_std ** 2 * (d ** 2 - d * dperp1 + dperp0 * (dperp0 - dperp1)) ** 2
        ) / ((d ** 2 + dperp0 ** 2 + dperp1 ** 2) ** 3 *
             (d ** 2 - d * (dperp0 + dperp1) + dperp0 ** 2 - dperp0 * dperp1 + dperp1 ** 2)))
