from mdt.utils import ComplexNoiseStdEstimator, create_roi
from mdt.exceptions import NoiseStdEstimationNotPossible
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2016-04-11"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AverageOfAir_ExtendedMask(ComplexNoiseStdEstimator):

    def estimate(self, problem_data, **kwargs):
        """Calculate the standard deviation of the error using the air (voxels outside the brain),

        This procedure first finds the extreme points of the given brain mask in all dimensions. Next, it extends
        this mask (as a sort cross) in all dimensions to mask out the mask and possible ghostings. Finally we mask the
        first N voxels at the edges of the brain since the may be zero-filled. We use all the remaining voxels for
        the noise std calculation.

        We then calculate per voxel the std of the noise and use that to estimate the noise of the original complex
        image domain using:
            sigma_complex = sqrt(2.0 / (4.0 - PI)) * stddev(signal in background region)

        Finally, we take the median value of all calculated std's.
        """
        voxels = self._get_air_voxels(problem_data)

        if not len(voxels):
            raise NoiseStdEstimationNotPossible('No voxels in air found.')

        return np.median(np.sqrt(2.0 / (4.0 - np.pi)) * np.std(voxels, axis=1))

    def _get_air_voxels(self, problem_data, border_offset=3):
        """Get a two dimensional list with all the voxels in the air.

        Returns:
            ndarray: The first dimension is the list of voxels, the second the signal per voxel.
        """
        indices = np.where(problem_data.mask > 0)
        max_dims = np.max(indices, axis=1)
        min_dims = np.min(indices, axis=1)

        mask = np.copy(problem_data.mask)

        mask[min_dims[0]:max_dims[0]] = True
        mask[:, min_dims[1]:max_dims[1], :] = True
        mask[..., min_dims[2]:max_dims[2]] = True

        mask[0:border_offset] = True
        mask[-border_offset:] = True
        mask[:, 0:border_offset, :] = True
        mask[:, -border_offset:, :] = True
        mask[..., 0:border_offset] = True
        mask[..., -border_offset:] = True

        return create_roi(problem_data.dwi_volume, np.invert(mask))

    def __str__(self):
        return __name__
