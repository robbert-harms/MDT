import numpy as np
from mdt import create_roi
from mdt.utils import ComplexNoiseStdEstimator
from mdt.exceptions import NoiseStdEstimationNotPossible

__author__ = 'Robbert Harms'
__date__ = "2015-11-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AllUnweightedVolumes(ComplexNoiseStdEstimator):

    def estimate(self, problem_data, **kwargs):
        """Calculate the standard deviation of the error using all unweighted volumes.

        This calculates per voxel (in the brain mas) the std over all unweighted volumes
        and takes the mean of those estimates as the standard deviation of the noise.

        The method is taken from Camino (http://camino.cs.ucl.ac.uk/index.php?n=Man.Estimatesnr).
        """
        unweighted_indices = problem_data.protocol.get_unweighted_indices()
        unweighted_volumes = problem_data.dwi_volume[..., unweighted_indices]

        if len(unweighted_indices) < 2:
            raise NoiseStdEstimationNotPossible('Not enough unweighted volumes for this estimator.')

        voxel_list = create_roi(unweighted_volumes, problem_data.mask)
        return np.mean(np.std(voxel_list, axis=1))

    def __str__(self):
        return __name__
