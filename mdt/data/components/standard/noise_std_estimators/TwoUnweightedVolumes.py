import numpy as np
from mdt import create_roi
from mdt.utils import ComplexNoiseStdEstimator
from mdt.exceptions import NoiseStdEstimationNotPossible

__author__ = 'Robbert Harms'
__date__ = "2015-11-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TwoUnweightedVolumes(ComplexNoiseStdEstimator):

    def estimate(self, problem_data, **kwargs):
        """Calculate the standard deviation of the error using the first two unweighted volumes/

        This subtracts the values of the first two unweighted volumes from each other, calculates the std over
        the results and divides that by sqrt(2).

        The method is taken from Camino (http://camino.cs.ucl.ac.uk/index.php?n=Man.Estimatesnr).

        Returns:
            float: single value representing the sigma for the given volume
        """
        unweighted_indices = problem_data.protocol.get_unweighted_indices()
        unweighted_volumes = problem_data.dwi_volume[..., unweighted_indices]

        if len(unweighted_indices) < 2:
            raise NoiseStdEstimationNotPossible('Not enough unweighted volumes for this estimator.')

        diff = unweighted_volumes[..., 0] - unweighted_volumes[..., 1]
        voxel_values = create_roi(diff, problem_data.mask)
        return np.std(voxel_values) / np.sqrt(2)

    def __str__(self):
        return __name__
