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
                    results['d.std'], results['dperp0.std'], results['dperp1.std'],
                    covariances=results.get('covariances', None)
                ),
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
        items = [
            ('MD', (results['d'] + results['dperp0'] + results['dperp1']) / 3.),
            ('FA', DTIMeasures.fractional_anisotropy(results['d'], results['dperp0'], results['dperp1'])),
            ('RD', (results['dperp0'] + results['dperp1']) / 2.0),
            ('AD', results['d']),
            ('d', results['d']),
            ('dperp0', results['dperp0']),
            ('dperp1', results['dperp1']),
        ]

        results = {}
        for name, data in items:
            results.update({name: np.mean(data, axis=1),
                            name + '.std': np.std(data, axis=1)})
        return results

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
    def fractional_anisotropy_std(d, dperp0, dperp1, d_std, dperp0_std, dperp1_std, covariances=None):
        """Calculate the standard deviation of the fractional anisotropy (FA) using error propagation.

        Args:
            d (ndarray): an 1d array
            dperp0 (ndarray): an 1d array
            dperp1 (ndarray): an 1d array
            d_std (ndarray): an 1d array
            dperp0_std (ndarray): an 1d array
            dperp1_std (ndarray): an 1d array
            covariances (dict): optionally, a matrix holding the covariances. This expects the keys to be like:
                '<param_0>_to_<param_1>'. The order of the parameter names does not matter.

        Returns:
            ndarray: the standard deviation of the fraction anisotropy using error propagation of the diffusivities.
        """
        gradient = DTIMeasures._get_fractional_anisotropy_gradient(d, dperp0, dperp1)
        covars = DTIMeasures._get_diffusivities_covariance_matrix(d_std, dperp0_std, dperp1_std,
                                                                  covariances=covariances)

        fa_std = np.zeros((gradient.shape[0]))
        for ind in range(gradient.shape[0]):
            fa_std[ind] = np.sqrt(np.dot(np.dot(gradient[ind], covars[ind]), gradient[ind]))
        return np.nan_to_num(fa_std)

    @staticmethod
    def _get_fractional_anisotropy_gradient(d, dperp0, dperp1):
        """Get the gradient of the Fractional Anisotropy function.

        This returns the gradient of the Fractional Anisotropy (FA) function, evaluated at the given diffusivities.
        This is required for error propagating the uncertainties of the diffusivities into FA. The gradient is given
        by the partial derivative of:

        .. math::

            \text{FA} = \sqrt{\frac{1}{2}} \frac{\sqrt{(d - d_{\perp_0})^2 + (d_{\perp_0} - d_{\perp_1})^2
                        + (d_{\perp_1} - d)^2}}{\sqrt{d^2 + d_{\perp_0}^2 + d_{\perp_1}^2}}

        Args:
            d (ndarray): an 1d vector with the principal diffusivity per voxel
            dperp0 (ndarray): an 1d vector with the first perpendicular diffusivity per voxel
            dperp1 (ndarray): an 1d vector with the second perpendicular diffusivity per voxel

        Returns:
            ndarray: a 2d vector with the gradient per voxel.
        """
        np.warnings.simplefilter("ignore")

        d, dperp0, dperp1 = (np.squeeze(el).astype(np.float64) for el in [d, dperp0, dperp1])

        return np.stack([
            (d ** 2 * (dperp0 + dperp1) + 2 * d * dperp0 * dperp1 - dperp0 ** 3
             - dperp0 ** 2 * dperp1 - dperp0 * dperp1 ** 2 - dperp1 ** 3)
            / (2 * (d ** 2 + dperp0 ** 2 + dperp1 ** 2) ** (3 / 2.)
               * np.sqrt(d ** 2 - d * (dperp0 + dperp1) + dperp0 ** 2 - dperp0 * dperp1 + dperp1 ** 2)),
            (-d ** 3 - d ** 2 * dperp1 + d * (dperp0 ** 2 + 2 * dperp0 * dperp1 - dperp1 ** 2) + dperp1 * (
            dperp0 ** 2 - dperp1 ** 2))
            / (2 * (d ** 2 + dperp0 ** 2 + dperp1 ** 2) ** (3 / 2.)
               * np.sqrt(d ** 2 - d * (dperp0 + dperp1) + dperp0 ** 2 - dperp0 * dperp1 + dperp1 ** 2)),
            (-d ** 3 - d ** 2 * dperp0 + d * (
            -dperp0 ** 2 + 2 * dperp0 * dperp1 + dperp1 ** 2) - dperp0 ** 3 + dperp0 * dperp1 ** 2)
            / (2 * (d ** 2 + dperp0 ** 2 + dperp1 ** 2) ** (3 / 2.)
               * np.sqrt(d ** 2 - d * (dperp0 + dperp1) + dperp0 ** 2 - dperp0 * dperp1 + dperp1 ** 2))
        ], axis=-1)

    @staticmethod
    def _get_diffusivities_covariance_matrix(d_std, dperp0_std, dperp1_std, covariances=None):
        """Get the covariance matrix of the diffusivities.

        This is required for the error propagation of the Fractional Anisotropy.
        """
        d_std, dperp0_std, dperp1_std = (np.squeeze(el) for el in [d_std, dperp0_std, dperp1_std])

        covars = np.zeros((d_std.shape[0], 3, 3)).astype(np.float64)
        covars[:, 0, 0] = d_std
        covars[:, 1, 1] = dperp0_std
        covars[:, 2, 2] = dperp1_std

        covars **= 2

        if covariances is not None:
            covars[:, 0, 1] = covars[:, 1, 0] = covariances.get('d_to_dperp0', covariances.get('dperp0_to_d', 0))
            covars[:, 0, 2] = covars[:, 2, 0] = covariances.get('d_to_dperp1', covariances.get('dperp1_to_d', 0))
            covars[:, 1, 2] = covars[:, 2, 1] = covariances.get('dperp0_to_dperp1',
                                                                covariances.get('dperp1_to_dperp0', 0))
        return covars
