import six
from mdt import load_dwi
from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
from mdt.data_loaders.protocol import autodetect_protocol_loader
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-10-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SigmaCalculator(object):

    def __init__(self, volume_info, protocol, mask):
        """Calculator for the standard deviation of the error.

        This is usually called sigma named after the use of this value in the Gaussian noise model.

        Args:
            volume_info (string or tuple): Either an (ndarray, img_header) tuple or the
                full path to the volume (4d signal data).
            protocol (Protocol or string): A protocol object with the right protocol for the given data,
                or a string object with a filename to the given file.
            brain_mask (string): A full path to a mask file that can optionally be used. If None given, no mask is used.
        """
        self._protocol = autodetect_protocol_loader(protocol).get_protocol()
        self._mask = autodetect_brain_mask_loader(mask).get_data()

        if isinstance(volume_info, six.string_types):
            self._signal4d, self._img_header = load_dwi(volume_info)
        else:
            self._signal4d, self._img_header = volume_info

    def calculate(self, **kwargs):
        """Calculate the sigma used in the evaluation models for the multi-compartment models.

        Returns:
            float: single value representing the sigma for the given volume
        """

    def get_used_rois(self, **kwargs):
        """Get a list of the ROI slices this classes uses.

        This can be used to visualize the ROIs.

        Returns:
            list of list of slices: per ROI a list of slices that constitute that ROI.
        """


class AverageOfAirROI(SigmaCalculator):

    def calculate(self, roi_size=2, **kwargs):
        """Calculate the standard deviation of the error using the air (voxels outside the brain),

        This will find the edges in the first dimension, the quarterpoints in the second and the middle slice in the
        last dimension to draw a ROI with the specified roi size.

        Next, it will append all the values of all the voxels and calculate the dot product of this vector to get the
        sum of squares. This will be divided by the length of the array to end up with a value for E(S^2).
        This follows the procedure in Camino in the file /src/apps/DataStats.java

        Finally we follow the procedure on the Camino website:
        (http://cmic.cs.ucl.ac.uk/camino/index.php?n=Man.Datastats)
        "
            Useful for estimating the noise level (as required for restore or mbalign) or signal to noise.
            An estimate of the noise level sigma (standard deviation of each component of the complex noise on
            the signal) is sqrt(E(S^2)/2) from an ROI entirely in background.
        "

        To end up with a return value for an estimate of the noise level.

        Args:
            roi_size (int): the size of the ROI's in all dimensions.
        """
        rois = self.get_used_rois(roi_size)
        voxel_values = np.array([self._signal4d[roi].flatten() for roi in rois]).flatten()

        sum_of_squares = np.dot(voxel_values, voxel_values)
        mean_squares = sum_of_squares / len(voxel_values)

        return np.sqrt(mean_squares / 2.0)

    def get_used_rois(self, roi_size=2):
        dist_from_edge = 5
        s = self._signal4d.shape
        return [[slice(dist_from_edge, 2*roi_size + dist_from_edge),
                 slice(s[1]//6 - roi_size, s[1]//6 + roi_size),
                 slice(s[2]//2 - roi_size, s[2]//2 + roi_size)],

                [slice(dist_from_edge, 2*roi_size + dist_from_edge),
                 slice(5 * s[1]//6 - roi_size, 5 * s[1]//6 + roi_size),
                 slice(s[2]//2 - roi_size, s[2]//2 + roi_size)],

                [slice(s[0]-2*roi_size-dist_from_edge, s[0] - dist_from_edge),
                 slice(s[1]//6 - roi_size, s[1]//6 + roi_size),
                 slice(s[2]//2 - roi_size, s[2]//2 + roi_size)],

                [slice(s[0]-2*roi_size-dist_from_edge, s[0] - dist_from_edge),
                 slice(5 * s[1]//6 - roi_size, 5 * s[1]//6 + roi_size),
                 slice(s[2]//2 - roi_size, s[2]//2 + roi_size)],
                ]