from mdt.utils import NoiseStdEstimator, NoiseStdEstimationNotPossible
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-11-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AverageOfAirROI(NoiseStdEstimator):

    def calculate(self, roi_size=2, **kwargs):
        """Calculate the standard deviation of the error using the air (voxels outside the brain),

        This will find the edges in the first dimension, the quarterpoints in the second dimension
        and the middle slice in the last dimension to draw a ROI with the specified roi size.

        Next, it will concatenate all the values of all the voxels and gradient directions and
        calculate the dot product of this vector to get the sum of squares.
        This will be divided by the length of the array to end up with a value for E(S^2).
        This follows the procedure in Camino in the file /src/apps/DataStats.java (Camino 2014)

        Finally we follow the procedure on the Camino website:
        (http://cmic.cs.ucl.ac.uk/camino/index.php?n=Man.Datastats)
        "
            An estimate of the noise level sigma (standard deviation of each component of the complex noise on
            the signal) is sqrt(E(S^2)/2) from an ROI entirely in background.
        "

        If the volume is zero filled we will raise an exception.

        Args:
            roi_size (int): the size of the ROI's in all dimensions.

        Raises:
            NoiseStdEstimationNotPossible: if the volume is zero filled
        """
        rois = self.get_used_rois(roi_size)
        voxel_values = np.array([self._signal4d[roi].flatten() for roi in rois]).flatten()

        sigma = np.sqrt(np.mean(np.power(voxel_values, 2)) / 2)
        if sigma < 1e-8:
            raise NoiseStdEstimationNotPossible('The volume appears to be zero filled, we can not use '
                                                'this sigma calculator on this volume.')
        return sigma

    def get_used_rois(self, roi_size=2):
        """Get a list of the ROI slices this classes uses.

        This can be used to visualize the ROIs.

        Args:
            roi_size (int): the size of the region of interest

        Returns:
            list of list of slices: per ROI a list of slices that constitute that ROI.
        """
        dist_from_edge = 5
        s = self._signal4d.shape
        return [[slice(dist_from_edge, 2 * roi_size + dist_from_edge),
                 slice(s[1] // 6 - roi_size, s[1] // 6 + roi_size),
                 slice(s[2] // 2 - roi_size, s[2] // 2 + roi_size)],

                [slice(dist_from_edge, 2 * roi_size + dist_from_edge),
                 slice(5 * s[1] // 6 - roi_size, 5 * s[1] // 6 + roi_size),
                 slice(s[2] // 2 - roi_size, s[2] // 2 + roi_size)],

                [slice(s[0] - 2 * roi_size - dist_from_edge, s[0] - dist_from_edge),
                 slice(s[1] // 6 - roi_size, s[1] // 6 + roi_size),
                 slice(s[2] // 2 - roi_size, s[2] // 2 + roi_size)],

                [slice(s[0] - 2 * roi_size - dist_from_edge, s[0] - dist_from_edge),
                 slice(5 * s[1] // 6 - roi_size, 5 * s[1] // 6 + roi_size),
                 slice(s[2] // 2 - roi_size, s[2] // 2 + roi_size)],
                ]
