from unittest import TestCase

from mot.cl_routines.sampling.metropolis_hastings import MetropolisHastings


__author__ = 'Robbert Harms'
__date__ = "2014-08-07"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TestMetropolis(TestCase):

    def test_sampling(self):
        sampler = MetropolisHastings()

        # assert(False)

