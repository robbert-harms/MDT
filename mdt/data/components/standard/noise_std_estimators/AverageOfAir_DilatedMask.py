from mdt.utils import ComplexNoiseStdEstimator, create_roi
from mdt.exceptions import NoiseStdEstimationNotPossible
import numpy as np
from scipy.ndimage.morphology import binary_dilation

__author__ = 'Robbert Harms'
__date__ = "2016-04-11"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AverageOfAir_DilatedMask(ComplexNoiseStdEstimator):

    def estimate(self, **kwargs):
        """Calculate the standard deviation of the error using the air (voxels outside the brain),

        This procedure first dilates the given brian mask a little bit to smooth out the edges. Finally we mask the
        first n voxels at the edges of the data volume since the may be zero-filled. We use all the remaining voxels for
        the noise std calculation.

        We then calculate per voxel the std of the noise and use that to estimate the noise of the original complex
        image domain using:
            sigma_complex = sqrt(2.0 / (4.0 - PI)) * stddev(signal in background region)

        Finally, we take the median value of all calculated std's.

        Raises:
            NoiseStdEstimationNotPossible: if the no voxels are left after the masking procedure
        """
        voxels = self._get_air_voxels()

        if not len(voxels):
            raise NoiseStdEstimationNotPossible('No voxels in air found.')

        return np.median(np.sqrt(2.0 / (4.0 - np.pi)) * np.std(voxels, axis=1))

    def _get_air_voxels(self, border_offset=3):
        """Get a two dimensional list with all the voxels in the air.

        Returns:
            ndarray: The first dimension is the list of voxels, the second the signal per voxel.
        """
        mask = np.copy(self._problem_data.mask)
        mask = binary_dilation(mask, iterations=1)

        mask[0:border_offset] = True
        mask[-border_offset:] = True
        mask[:, 0:border_offset, :] = True
        mask[:, -border_offset:, :] = True
        mask[..., 0:border_offset] = True
        mask[..., -border_offset:] = True

        return create_roi(self._problem_data.dwi_volume, np.invert(mask))
