import six
from mdt import load_dwi
from mdt.data_loader.brain_mask import autodetect_brain_mask_loader
from mdt.data_loader.protocol import autodetect_protocol_loader
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-10-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SigmaCalculator(object):

    def __init__(self, volume_info, protocol, mask):
        """Calculator for the standard deviation of the error

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

        Args:
            roi_size (int): the size of the ROI's in all dimensions.
        """
        rois = self.get_used_rois(roi_size)
        voxel_values = np.array([self._signal4d[roi].flatten() for roi in rois])

        print(voxel_values)
        print(np.mean(voxel_values))
        print(np.sqrt(np.mean(np.square(voxel_values))))

        for roi in rois:
            self._signal4d[roi] = 1e4

        from mdt import view_results_slice
        view_results_slice({'signal4d': self._signal4d})

        return np.std(voxel_values)

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