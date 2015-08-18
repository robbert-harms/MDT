from unittest import TestCase
import os
import itertools

import scipy.io
from numpy.testing import assert_array_equal

from mot.utils import apply_mask, restore_roi


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TestDMRI(TestCase):
    def setUp(self):
        single_slice = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                                                        'includes', 'test_data',
                                                                        'skyra', 'single_slice.mat')))
        single_slice_roi = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                                                         'includes', 'test_data', 'skyra', 
                                                                         'single_slice_roi.mat')))
        self.signal4d = single_slice['signal4d']
        self.brain_mask = single_slice['brain_mask'].astype(bool)
        self.signals = single_slice_roi['signals']
        self.idx = single_slice_roi['idx']
        self.gen_signals = create_roi(self.signal4d, self.brain_mask)
        self.restored_signal4d = restore_roi(self.signals, self.brain_mask)

    def test_createROI_signals(self):
        assert_array_equal(self.signals, self.gen_signals)

    def test_restoreROI(self):
        assert_array_equal(self.signal4d.shape, self.restored_signal4d.shape)

        try:
            xrange
        except NameError:
            xrange = range

        for x, y, z in itertools.product(*map(xrange, self.signal4d.shape[0:3])):
            if self.brain_mask[x, y, z] > 0:
                assert_array_equal(self.signal4d[x, y, z, :], self.restored_signal4d[x, y, z, :])