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

        problem_data = mdt.load_problem_data(pjoin('b1k_b2k_example_slices_24_38'),
                                             pjoin('b1k_b2k.prtcl'),
                                             pjoin('b1k_b2k_example_slices_24_38_mask'))

        for model_name in ['BallStick_r1 (Cascade)', 'Tensor (Cascade)', 'NODDI (Cascade)']:
            mdt.fit_model(model_name, problem_data, pjoin('output', 'b1k_b2k_example_slices_24_38_mask'))

    @classmethod
    def _run_b6k_analysis(cls):
        pjoin = mdt.make_path_joiner(os.path.join(cls._tmp_dir, cls._tmp_dir_subdir, 'multishell_b6k_max'))

        problem_data = mdt.load_problem_data(pjoin('multishell_b6k_max_example_slices_24_38'),
                                             pjoin('multishell_b6k_max.prtcl'),
                                             pjoin('multishell_b6k_max_example_slices_24_38_mask'))

        for model_name in ['CHARMED_r1 (Cascade|fixed)', 'CHARMED_r2 (Cascade|fixed)', 'CHARMED_r3 (Cascade|fixed)']:
            mdt.fit_model(model_name, problem_data, pjoin('output', 'multishell_b6k_max_example_slices_24_38_mask'))

    def test_b1k_b2k_ballstick(self):
        pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'b1k_b2k'))

        known_volumes = mdt.load_volume_maps(pjoin('test_output', 'b1k_b2k_example_slices_24_38_mask', 'BallStick_r1'))
        user_volumes = mdt.load_volume_maps(pjoin('output', 'b1k_b2k_example_slices_24_38_mask', 'BallStick_r1'))

        msg_prefix = 'b1k_b2k - BallStick_r1'

        self._test_map(user_volumes, known_volumes, 'LogLikelihood', msg_prefix)
        self._test_map(user_volumes, known_volumes, 'w_stick.w', msg_prefix)
        self._test_weighted_maps(user_volumes, known_volumes, ['Stick.theta', 'Stick.phi'],
                                 'w_stick.w', msg_prefix)

    def test_b1k_b2k_tensor(self):
        pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'b1k_b2k'))

        known_volumes = mdt.load_volume_maps(pjoin('test_output', 'b1k_b2k_example_slices_24_38_mask', 'Tensor'))
        user_volumes = mdt.load_volume_maps(pjoin('output', 'b1k_b2k_example_slices_24_38_mask', 'Tensor'))

        msg_prefix = 'b1k_b2k - Tensor'

        for map_name in known_volumes:
            self._test_map(user_volumes, known_volumes, map_name, msg_prefix)

    def test_b6k_charmed_r1(self):
        pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'multishell_b6k_max'))

        known_volumes = mdt.load_volume_maps(pjoin('test_output', 'multishell_b6k_max_example_slices_24_38_mask',
                                                   'CHARMED_r1'))
        user_volumes = mdt.load_volume_maps(pjoin('output', 'multishell_b6k_max_example_slices_24_38_mask',
                                                  'CHARMED_r1'))

        msg_prefix = 'b6k_max - CHARMED_r1'

        for map_name in known_volumes:
            self._test_map(user_volumes, known_volumes, map_name, msg_prefix)

    def test_b6k_charmed_r2(self):
        pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'multishell_b6k_max'))

        known_volumes = mdt.load_volume_maps(pjoin('test_output', 'multishell_b6k_max_example_slices_24_38_mask',
                                                   'CHARMED_r2'))
        user_volumes = mdt.load_volume_maps(pjoin('output', 'multishell_b6k_max_example_slices_24_38_mask',
                                                  'CHARMED_r2'))

        msg_prefix = 'b6k_max - CHARMED_r2'

        for map_name in known_volumes:
            self._test_map(user_volumes, known_volumes, map_name, msg_prefix)

    def test_b6k_charmed_r3(self):
        pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'multishell_b6k_max'))

        known_volumes = mdt.load_volume_maps(pjoin('test_output', 'multishell_b6k_max_example_slices_24_38_mask',
                                                   'CHARMED_r2'))
        user_volumes = mdt.load_volume_maps(pjoin('output', 'multishell_b6k_max_example_slices_24_38_mask',
                                                  'CHARMED_r2'))

        msg_prefix = 'b6k_max - CHARMED_r3'

        for map_name in known_volumes:
            self._test_map(user_volumes, known_volumes, map_name, msg_prefix)

    def _test_map(self, user_volumes, known_volumes, map_to_test, msg_prefix, rtol=1e-4):
        np.testing.assert_allclose(np.mean(user_volumes[map_to_test]), np.mean(known_volumes[map_to_test]),
                                   rtol=rtol, err_msg='{} - {} - mean'.format(map_to_test, msg_prefix))
        np.testing.assert_allclose(np.std(user_volumes[map_to_test]), np.std(known_volumes[map_to_test]),
                                   rtol=rtol, err_msg='{} - {} - std'.format(map_to_test, msg_prefix))

    def _test_weighted_maps(self, user_volumes, known_volumes, maps_to_test, map_to_weight_by, msg_prefix, rtol=1e-4):
        for map_name in maps_to_test:
            known = known_volumes[map_name] * known_volumes[map_to_weight_by]
            user = user_volumes[map_name] * user_volumes[map_to_weight_by]

            np.testing.assert_allclose(np.mean(user), np.mean(known),
                                       rtol=rtol, err_msg='{} - {} - mean'.format(msg_prefix, map_name))
            np.testing.assert_allclose(np.std(user), np.std(known),
                                       rtol=rtol, err_msg='{} - {} - std'.format(msg_prefix, map_name))


if __name__ == '__main__':
    unittest.main()
