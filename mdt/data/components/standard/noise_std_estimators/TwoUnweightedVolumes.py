import numpy as np
from mdt import create_roi, create_median_otsu_brain_mask
from mdt.utils import ComplexNoiseStdEstimator, NoiseStdEstimationNotPossible

__author__ = 'Robbert Harms'
__date__ = "2015-11-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TwoUnweightedVolumes(ComplexNoiseStdEstimator):

    def estimate(self, **kwargs):
        """Calculate the standard deviation of the error using the first two unweighted volumes/

        This subtracts the values of the first two unweighted volumes from each other, calculates the std over
        the results and divides that by sqrt(2).

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

        diff = unweighted_volumes[..., 0] - unweighted_volumes[..., 1]
        voxel_values = create_roi(diff, self._mask)
        return np.std(voxel_values) / np.sqrt(2)

