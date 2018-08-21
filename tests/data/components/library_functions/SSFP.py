import unittest
import numpy as np
from numpy import cos, exp, sin
from numpy.testing import assert_allclose
import mdt
from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import Array


class test_SSFP(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_ssfp_float(self):
        test_params = self._generate_test_params().astype(dtype=np.float32)

        python_results = self._calculate_python(test_params)
        cl_results = self._calculate_cl(test_params, double_precision=False)

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-5)

    def test_ssfp_double(self):
        test_params = self._generate_test_params().astype(dtype=np.float64)

        python_results = self._calculate_python(test_params)
        cl_results = self._calculate_cl(test_params, double_precision=True)

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-7)

    def _calculate_cl(self, test_params, double_precision=False):
        ssfp = mdt.lib.components.get_component('library_functions', 'SSFP')()

        names = ['d', 'delta', 'G', 'TR', 'flip_angle', 'b1', 'T1', 'T2']
        input_data = dict(zip(names, [Array(test_params[..., ind], as_scalar=True)
                                      for ind in range(test_params.shape[1])]))

        return ssfp.evaluate(input_data, test_params.shape[0],
                             cl_runtime_info=CLRuntimeInfo(double_precision=double_precision))

    def _generate_test_params(self):
        """

        [d (m/s^2), delta (s), G (T/m), TR (s), flip_angle (rad), b1 (a.u.), T1 (s), T2 (s)]

        """
        test_param_sets = [
            # In vivo
            {'default': [1e-9, 1e-2, 5e-2, 0.5, np.pi / 6, 1.0, 1.5, 1.0],
             'lower_bounds': [0, 1e-5, 1e-5, 1e-4, 0, 0.1, 1e-5, 1e-5],
             'upper_bounds': [1e-8, 1e-1, 1e-1, 2, np.pi / 2, 2.0, 4, 2]},
            # Ex vivo
            {'default': [1e-10, 1e-2, 5e-2, 0.05, np.pi / 6, 1.0, 0.5, 0.05],
             'lower_bounds': [0, 1e-5, 1e-5, 1e-4, 0, 0.1, 1e-3, 1e-3],
             'upper_bounds': [1e-8, 1e-1, 1e-1, 0.1, np.pi / 2, 2.0, 1.0, 0.1]}]

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

        # filter delta >= TR
        test_cases = test_cases[test_cases[:, 1] < test_cases[:, 3]]

        return test_cases

    def _calculate_python(self, input_params):
        results = np.zeros(input_params.shape[0])

        for ind in range(input_params.shape[0]):
            results[ind] = self._ssfp_python(*list(input_params[ind, :]))

        return results

    def _ssfp_python(self, d, delta, G, TR, flip_angle, b1, T1, T2):
        larmor_freq = 267.5987e6
        alpha = flip_angle * b1

        E1 = exp(-TR / T1)
        E2 = exp(-TR / T2)

        b = (larmor_freq * G * delta)**2 * TR
        beta = (larmor_freq * G * delta)**2 * delta

        A1 = exp(-b * d)
        A2 = exp(-beta * d)

        s = E2 * A1 * A2**(-4/3.0) * (1 - E1 * cos(alpha)) + E2 * A2**(-1/3.0) * (cos(alpha) - 1)

        r = 1 - E1 * cos(alpha) + E2**2 * A1 * A2**(1/3.0) * (cos(alpha) - E1)

        if (1 - E1 * A1) < 1e-10:
            print(dict(d=d, delta=delta, G=G, TR=TR, flip_angle=flip_angle, b1=b1, T1=T1, T2=T2))

        K = (1 - E1 * A1 * cos(alpha) - E2**2 * A1**2 * A2**(-2/3.0) * (E1 * A1 - cos(alpha))) \
            / (E2 * A1 * A2**(-4/3.0) * (1 + cos(alpha)) * (1 - E1 * A1))

        F1 = K - np.sqrt(K**2 - A2**2)

        return -((1 - E1) * E2 * A2**(-2/3.0) * (F1 - E2 * A1 * A2**(2/3.0)) * sin(alpha)) / (r - F1 * s)
