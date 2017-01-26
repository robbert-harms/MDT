#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mot
----------------------------------

Tests for `mot` module.
"""
import tempfile
import unittest
import numpy as np
import shutil
import mdt
import os
from pkg_resources import resource_filename

from mdt.utils import split_image_path


class ExampleDataTest(unittest.TestCase):

    def __init__(self, *args):
        super(ExampleDataTest, self).__init__(*args)
        self._tmp_dir = None
        self._tmp_dir_subdir = 'mdt_example_data'

    def setUp(self):
        if not self._tmp_dir:
            self._tmp_dir = tempfile.mkdtemp('mdt_example_data_test')
        shutil.copytree(os.path.abspath(resource_filename('mdt', 'data/mdt_example_data')),
                        os.path.join(self._tmp_dir, self._tmp_dir_subdir))

    def tearDown(self):
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir)

    def test_b1k_b2k(self):
        pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'b1k_b2k'))

        problem_data = mdt.load_problem_data(pjoin('b1k_b2k_example_slices_24_38'),
                                             pjoin('b1k_b2k.prtcl'),
                                             pjoin('b1k_b2k_example_slices_24_38_mask'))

        output_dir = pjoin('output', 'b1k_b2k_example_slices_24_38_mask')

        models_to_test = [('S0', 'S0'),
                          ('BallStick_r1 (Cascade)', 'BallStick_r1'),
                          ('NODDI (Cascade)', 'NODDI'),
                          ('Tensor (Cascade)', 'Tensor')]

        for model_name, model_basename in models_to_test:
            mdt.fit_model(model_name, problem_data, output_dir)
            self._assert_compare_maps(os.path.join(output_dir, model_basename),
                                      pjoin('test_output', 'b1k_b2k_example_slices_24_38_mask', model_basename,),
                                      'b1k_b2k - {}'.format(model_basename))

    def test_b6k(self):
        pjoin = mdt.make_path_joiner(os.path.join(self._tmp_dir, self._tmp_dir_subdir, 'b6k'))

        problem_data = mdt.load_problem_data(pjoin('b6k_example_slices_24_38'),
                                             pjoin('b6k.prtcl'),
                                             pjoin('b6k_example_slices_24_38_mask'))

        output_dir = pjoin('output', 'b6k_example_slices_24_38_mask')

        models_to_test = [('S0', 'S0'),
                          ('BallStick_r1 (Cascade)', 'BallStick_r1'),
                          ('BallStick_r2 (Cascade)', 'BallStick_r2'),
                          ('BallStick_r3 (Cascade)', 'BallStick_r3'),
                          ('CHARMED_r1 (Cascade)', 'CHARMED_r1'),
                          ('CHARMED_r2 (Cascade)', 'CHARMED_r2'),
                          ('CHARMED_r3 (Cascade)', 'CHARMED_r3')
                          ]

        for model_name, model_basename in models_to_test:
            mdt.fit_model(model_name, problem_data, output_dir)
            self._assert_compare_maps(os.path.join(output_dir, model_basename),
                                      pjoin('test_output', 'b6k_example_slices_24_38_mask', model_basename),
                                      'b1k_b2k - {}'.format(model_basename))

    def _assert_compare_maps(self, user_maps_dir, known_maps_dir, error_msg_prefix):
        """Test the user calculated maps against the known good maps.

        Args:
            user_maps_dir (str): the directory containing the maps calculated by the user as part of this test
            known_maps_dir (str): the directory containing the well known good maps
            error_msg_prefix (str): the prefix for the error message
        """
        for test_map in os.listdir(known_maps_dir):
            map_name = split_image_path(test_map)[1]

            user_map = mdt.load_nifti(os.path.join(user_maps_dir, test_map)).get_data()
            known_map = mdt.load_nifti(os.path.join(known_maps_dir, test_map)).get_data()

            np.testing.assert_allclose(np.mean(user_map), np.mean(known_map),
                                       rtol=0.1,
                                       err_msg=error_msg_prefix + '- {} - mean'.format(map_name))
            np.testing.assert_allclose(np.std(user_map), np.std(known_map),
                                       rtol=0.1,
                                       err_msg=error_msg_prefix + ' - {} - std'.format(map_name))


if __name__ == '__main__':
    unittest.main()
