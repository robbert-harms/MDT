import itertools
import numpy as np
from mot.statistics import fit_multivariate_gaussian, TruncatedGaussian, CircularGaussian, \
    StandardGaussian

__author__ = 'Robbert Harms'
__date__ = '2017-11-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class MCMCDistributionFit(object):

    def set_parameter_options(self, param_name, options, overwrite=True):
        """A catch-all method for configuring the distribution per parameter.

        Args:
            param_name (str): the parameter name to set the options for
            options (dict): the options to set
            overwrite (boolean): if we want to overwrite the values if they are already present
        """
        raise NotImplementedError()

    def fit_samples(self, samples, param_names, lower_bounds, upper_bounds):
        """Fit the distribution to the samples.

        This should return a dictionary with a selection of 2d or 3d matrices that contain the parameters of the fitted
        model per voxel.

        Args:
            samples (ndarray): the 3d matrix with the samples
            param_names (list of str): the list with the parameter names of each of the sampled parameters
            lower_bounds (list): for each parameter the lower bounds
            upper_bounds (list): for each parameter the upper bounds

        Returns:
            dict: a dictionary with maps to write as result of fitting this distribution
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.fit_samples(*args, **kwargs)


class MultivariateGaussian(MCMCDistributionFit):

    def __init__(self, distribution_types=None):
        """Fit a multivariate Gaussian to your samples.

        The mean of a multivariate Gaussian is identical to the list of means of each of the marginals and the
        covariance matrix can be calculated using the sample covariance between any pair of parameters. Based on that,
        we can easily allow for different distribution types for each of the parameters (like circular and truncated
        Gaussians) by using those instead of a regular Gaussian for each of the marginals.

        The distribution types provided are either a string or implementations of
        :class:`mot.statistics.UnimodalGaussianType`. If a string is given it is one of ``circular``, ``truncated``
        or ``standard``. In the case of ``circular`` we assume a circular Gaussian around the upper and lower bounds.
        If ``truncated`` we truncated the distribution around the upper and lower bounds and ``standard`` means we
        apply a typical Gaussian.

        Args:
            distribution_types (dict): the type of unimodal Gaussian we can use for the marginal of that parameter.
                If not set we use a standard Gaussian.
        """
        self._distribution_types = distribution_types or {}

    def set_parameter_options(self, param_name, options, overwrite=True):
        if 'distribution_type' in options:
            if param_name in self._distribution_types:
                if overwrite:
                    self._distribution_types[param_name] = options['distribution_type']
            else:
                self._distribution_types[param_name] = options['distribution_type']

    def fit_samples(self, samples, param_names, lower_bounds, upper_bounds):
        distribution_types = self._get_distribution_types(param_names)
        means, covars = fit_multivariate_gaussian(samples, lower_bounds, upper_bounds, distribution_types)
        return self._results_to_maps(means, covars, param_names)

    def _get_distribution_types(self, param_names):
        distribution_types = []
        for param_name in param_names:
            if param_name in self._distribution_types:
                distribution_types.append(self._resolve_distribution_type(self._distribution_types[param_name]))
            else:
                distribution_types.append(StandardGaussian())
        return distribution_types

    def _resolve_distribution_type(self, distribution_type):
        if distribution_type == 'standard':
            return StandardGaussian()
        if distribution_type == 'circular':
            return CircularGaussian()
        if distribution_type == 'truncated':
            return TruncatedGaussian()
        return distribution_type

    def _results_to_maps(self, means, covariances, param_names):
        """Converts the raw results into interpretable maps."""
        results = {}

        for ind0, ind1 in itertools.product(range(len(param_names)), range(len(param_names))):
            param_name0 = param_names[ind0]
            param_name1 = param_names[ind1]

            if ind0 == ind1:
                results['{}.std'.format(param_name0)] = np.sqrt(covariances[:, ind0, ind1])
            else:
                results['Covariance_{}_to_{}'.format(param_name0, param_name1)] = covariances[:, ind0, ind1]

        for ind, param_name in enumerate(param_names):
            results[param_name] = means[:, ind]

        return results
