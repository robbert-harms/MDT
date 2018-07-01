#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mdt
----------------------------------

Tests for `mdt` module.
"""
import tempfile
import unittest
import numpy as np
import shutil
import mdt
import os
from pkg_resources import resource_filename


class ExampleDataTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp('mdt_example_data_test')
        cls._tmp_dir_subdir = 'mdt_example_data'

        shutil.copytree(os.path.abspath(resource_filename('mdt', 'data/mdt_example_data')),
                        os.path.join(cls._tmp_dir, cls._tmp_dir_subdir))

        cls._run_b1k_b2k_analysis()
        cls._run_b6k_analysis()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp_dir)

    @classmethod
    def _run_b1k_b2k_analysis(cls):
        pjoin = mdt.make_path_joiner(os.path.join(cls._tmp_dir, cls._tmp_dir_subdir, 'b1k_b2k'))

        input_data = mdt.load_input_data(pjoin('b1k_b2k_example_slices_24_38'),
                                         pjoin('b1k_b2k.prtcl'),
                                         pjoin('b1k_b2k_example_slices_24_38_mask'))

        for model_name in ['BallStick_r1 (Cascade)', 'Tensor (Cascade)', 'NODDI (Cascade|fixed)']:
            mdt.fit_model(model_name, input_data, pjoin('output', 'b1k_b2k_example_slices_24_38_mask'))

    @classmethod
    def _run_b6k_analysis(cls):
        pjoin = mdt.make_path_joiner(os.path.join(cls._tmp_dir, cls._tmp_dir_subdir, 'multishell_b6k_max'))

        input_data = mdt.load_input_data(pjoin('multishell_b6k_max_example_slices_24_38'),
                                         pjoin('multishell_b6k_max.prtcl'),
                                         pjoin('multishell_b6k_max_example_slices_24_38_mask'))

        for model_name in ['CHARMED_r1 (Cascade|fixed)', 'CHARMED_r2 (Cascade|fixed)', 'CHARMED_r3 (Cascade|fixed)']:
            mdt.fit_model(model_name, input_data, pjoin('output', 'multishell_b6k_max_example_slices_24_38_mask'))

    def test_lls_b1k_b2k(self):
        known_values = {
            'BallStick_r1':
                {'LogLikelihood':
                     {'mean': -1215.5128173828125, 'std': 924.2643432617188}},
            'Tensor':
                {'LogLikelihood':
                     {'mean': -182.73013305664062, 'std': 20.022197723388672}},
            'NODDI':
                {'LogLikelihood':
                     {'mean': -451.26239013671875, 'std': 37.696372985839844}}}

        for model_name in ['BallStick_r1', 'Tensor', 'NODDI']:
            pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'b1k_b2k'))

            user_volumes = mdt.load_volume_maps(pjoin('output', 'b1k_b2k_example_slices_24_38_mask', model_name))

            msg_prefix = 'b1k_b2k - {}'.format(model_name)
            roi = mdt.create_roi(user_volumes['LogLikelihood'], pjoin('b1k_b2k_example_slices_24_38_mask'))

            for map_name, test_values in known_values[model_name].items():
                np.testing.assert_allclose(test_values['mean'], np.mean(roi),
                                           rtol=1e-4, err_msg='{} - {} - mean'.format(msg_prefix, map_name))
                np.testing.assert_allclose(test_values['std'], np.std(roi),
                                           rtol=1e-4, err_msg='{} - {} - std'.format(msg_prefix, map_name))

    def test_lls_multishell_b6k_max(self):
        known_values = {
            'CHARMED_r1':
                {'LogLikelihood':
                     {'mean': -442.6887512207031, 'std': 31.19772720336914}},
            'CHARMED_r2':
                {'LogLikelihood':
                     {'mean': -436.6225280761719, 'std': 27.23966407775879}},
            'CHARMED_r3':
                {'LogLikelihood':
                     {'mean': -433.6161804199219, 'std': 24.84543228149414}}}

        for model_name in ['CHARMED_r1', 'CHARMED_r2', 'CHARMED_r3']:
            pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'multishell_b6k_max'))

            user_volumes = mdt.load_volume_maps(
                pjoin('output', 'multishell_b6k_max_example_slices_24_38_mask', model_name))

            msg_prefix = 'b1k_b2k - {}'.format(model_name)
            roi = mdt.create_roi(user_volumes['LogLikelihood'], pjoin('multishell_b6k_max_example_slices_24_38_mask'))

            for map_name, test_values in known_values[model_name].items():
                np.testing.assert_allclose(test_values['mean'], np.mean(roi),
                                           rtol=1e-4, err_msg='{} - {} - mean'.format(msg_prefix, map_name))
                np.testing.assert_allclose(test_values['std'], np.std(roi),
                                           rtol=1e-4, err_msg='{} - {} - std'.format(msg_prefix, map_name))


if __name__ == '__main__':
    unittest.main()
