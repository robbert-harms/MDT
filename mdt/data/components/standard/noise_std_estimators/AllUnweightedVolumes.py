import numpy as np
from mdt import create_roi, create_median_otsu_brain_mask
from mdt.utils import ComplexNoiseStdEstimator, NoiseStdEstimationNotPossible

__author__ = 'Robbert Harms'
__date__ = "2015-11-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AllUnweightedVolumes(ComplexNoiseStdEstimator):

    def estimate(self, **kwargs):
        """Calculate the standard deviation of the error using all unweighted volumes.

        This calculates per voxel the std over all unweighted volumes and takes the mean of those estimates as
        the standard deviation of the noise.

        The method is taken from Camino (http://camino.cs.ucl.ac.uk/index.php?n=Man.Estimatesnr).

        Returns:
            float: single value representing the sigma for the given volume
        """
        if self._mask is None:
            self._mask = create_median_otsu_brain_mask(self._signal4d, self._protocol)

        unweighted_indices = self._protocol.get_unweighted_indices()
        unweighted_volumes = self._signal4d[..., unweighted_indices]

        if len(unweighted_volumes) < 2:
            raise NoiseStdEstimationNotPossible('Not enough unweighted volumes for this estimator.')

        voxel_list = create_roi(unweighted_volumes, self._mask)
        return np.mean(np.std(voxel_list, axis=1))
