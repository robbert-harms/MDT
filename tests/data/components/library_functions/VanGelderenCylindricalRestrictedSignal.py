import unittest
from scipy.special import jnp_zeros
import numpy as np
from numpy.testing import assert_allclose
import mdt
from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import Array


class test_VanGelderenCylindricalRestrictedSignal(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_ncrs_float(self):
        test_params = self._generate_test_params().astype(dtype=np.float32)

        python_results = self._calculate_python(test_params)
        cl_results = self._calculate_cl(test_params, double_precision=False)

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-5, rtol=1e-5)

    def test_ncrs_double(self):
        test_params = self._generate_test_params().astype(dtype=np.float64)

        python_results = self._calculate_python(test_params)
        cl_results = self._calculate_cl(test_params, double_precision=True)

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-7)

    def _calculate_cl(self, test_params, double_precision=False):
        func = mdt.lib.components.get_component('library_functions', 'VanGelderenCylinder')()

        names = ['Delta', 'delta', 'd', 'R', 'G']
        input_data = dict(zip(names, [Array(test_params[..., ind], as_scalar=True)
                                      for ind in range(test_params.shape[1])]))

        return func.evaluate(input_data, test_params.shape[0],
                             cl_runtime_info=CLRuntimeInfo(double_precision=double_precision))

    def _generate_test_params(self):
        """

        [Delta (s), delta (s), d (m/s^2), R (m), G (T/m)]

        """
        test_param_sets = [
            {'default': [0.5, 1e-2, 1e-9, 1e-6, 0.05],
             'lower_bounds': [0.1, 1e-3, 1e-10, 1e-7, 1e-4],
             'upper_bounds': [1, 0.1, 1e-8, 2e-5, 0.1]}
        ]

        def generate_params_matrix(defaults, lower_bounds, upper_bounds, nmr_steps):
            params_matrix = np.tile(default_values, (nmr_steps * len(default_values), 1))

            for ind in range(len(default_values)):
                params_matrix[(ind * nmr_steps):((ind + 1) * nmr_steps), ind] = \
                    np.linspace(lower_bounds[ind], upper_bounds[ind], num=nmr_steps)

            return params_matrix

        nmr_steps = 100

        matrices = []

        for param_set in test_param_sets:
            default_values = param_set['default']
            lower_bounds = param_set['lower_bounds']
            upper_bounds = param_set['upper_bounds']

            matrices.append(generate_params_matrix(default_values, lower_bounds, upper_bounds, nmr_steps))

        test_cases = np.vstack(matrices)

        return test_cases

    def _calculate_python(self, input_params):
        results = np.zeros(input_params.shape[0])

        for ind in range(input_params.shape[0]):
            results[ind] = self._ncrs_python(*list(input_params[ind, :]))

        return results

    def _ncrs_python(self, Delta, delta, d, R, G):
        if R == 0 or R < np.finfo(float).eps:
            return 0

        GAMMA = 267.5987E6
        alpha_roots = jnp_zeros(1, 16) / R

        sum = 0
        for i in range(alpha_roots.shape[0]):
            alpha = alpha_roots[i]

            num = (2 * d * alpha**2 * delta
                   - 2
                   + 2 * np.exp(-d * alpha**2 * delta)
                   + 2 * np.exp(-d * alpha**2 * Delta)
                   - np.exp(-d * alpha**2 * (Delta - delta))
                   - np.exp(-d * alpha**2 * (Delta + delta)))
            dem = d**2 * alpha**6 * (R**2 * alpha**2 - 1)

            sum += (num / dem)

        return -2 * GAMMA**2 * G**2 * sum
