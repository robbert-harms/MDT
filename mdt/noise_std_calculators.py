import logging
import tempfile
import shutil
import six
from mdt import load_dwi, fit_model, restore_volumes, create_roi, create_median_otsu_brain_mask
from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
from mdt.data_loaders.protocol import autodetect_protocol_loader
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-10-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NoiseStdCalculator(object):

    def __init__(self, volume_info, protocol, mask=None):
        """Calculator for the standard deviation of the error.

        This is usually called sigma named after the use of this value in the Gaussian noise model.

        Args:
            volume_info (string or tuple): Either an (ndarray, img_header) tuple or the
                full path to the volume (4d signal data).
            protocol (Protocol or string): A protocol object with the right protocol for the given data,
                or a string object with a filename to the given file.
            brain_mask (string): A full path to a mask file that can optionally be used.
                If None given, we will create one if necessary.
        """
        self._volume_info = volume_info
        self._protocol = autodetect_protocol_loader(protocol).get_protocol()
        self._logger = logging.getLogger(__name__)

        if mask is not None:
            self._mask = autodetect_brain_mask_loader(mask).get_data()
        else:
            self._mask = None

        if isinstance(volume_info, six.string_types):
            self._signal4d, self._img_header = load_dwi(volume_info)
        else:
            self._signal4d, self._img_header = volume_info

    def calculate(self, **kwargs):
        """Calculate the sigma used in the evaluation models for the multi-compartment models.

        Returns:
            float: single value representing the sigma for the given volume

        Raises:
            ValueError: if we can not calculate the sigma using this calculator an exception is raised.
        """


class AverageOfAirROI(NoiseStdCalculator):

    def calculate(self, roi_size=2, **kwargs):
        """Calculate the standard deviation of the error using the air (voxels outside the brain),

        This will find the edges in the first dimension, the quarterpoints in the second and the middle slice in the
        last dimension to draw a ROI with the specified roi size.

        Next, it will append all the values of all the voxels and calculate the dot product of this vector to get the
        sum of squares. This will be divided by the length of the array to end up with a value for E(S^2).
        This follows the procedure in Camino in the file /src/apps/DataStats.java (Camino 2014)

        Finally we follow the procedure on the Camino website:
        (http://cmic.cs.ucl.ac.uk/camino/index.php?n=Man.Datastats)
        "
            Useful for estimating the noise level (as required for restore or mbalign) or signal to noise.
            An estimate of the noise level sigma (standard deviation of each component of the complex noise on
            the signal) is sqrt(E(S^2)/2) from an ROI entirely in background.
        "

        To end up with a return value for an estimate of the noise level.

        If the volume is zero filled we will raise an exception.

        Args:
            roi_size (int): the size of the ROI's in all dimensions.

        Raises:
            ValueError: if the volume is zero filled we would have returned a value of 0.0, instead we raise an error.
        """
        rois = self.get_used_rois(roi_size)
        voxel_values = np.array([self._signal4d[roi].flatten() for roi in rois]).flatten()

        sum_of_squares = np.dot(voxel_values, voxel_values)
        mean_squares = sum_of_squares / len(voxel_values)

        sigma = np.sqrt(mean_squares / 2.0)
        if sigma < 1e-8:
            raise ValueError('The volume is zero filled we can not use this sigma calculator on this volume.')
        return sigma

    def get_used_rois(self, roi_size=2):
        """Get a list of the ROI slices this classes uses.

        This can be used to visualize the ROIs.

        Returns:
        list of list of slices: per ROI a list of slices that constitute that ROI.
        """
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


class AverageOfUnweightedVolumes(NoiseStdCalculator):

    def calculate(self, **kwargs):
        """Calculate the standard deviation of the error using the unweighted volumes.

        We first fit a S0 model to the data, and subtract this estimate from the unweighted volumes. Next, we
        compute per voxel the sum of squares divided by the number of voxels. This is the E[S^2] per voxel. We then
        take for each voxel the sqrt(E[S^2]) as estimate for the noise std in that voxel.

        Finally, we calculate the mean of all those voxels.
        """
        if self._mask is None:
            self._mask = create_median_otsu_brain_mask(self._volume_info, self._protocol)

        s0_vol = self._get_s0_fit()
        unweighted_indices = self._protocol.get_unweighted_indices()
        unweighted_volumes = self._signal4d[..., unweighted_indices]
        baseline_images = unweighted_volumes - s0_vol
        voxel_values = create_roi(baseline_images, self._mask)

        sum_of_squares = np.sum(np.power(voxel_values, 2), axis=1)
        mean_squares = sum_of_squares / voxel_values.shape[0]

        sigmas = np.sqrt(mean_squares / 2.0)
        return np.mean(sigmas)

    def _get_s0_fit(self):
        self._logger.info('Estimating S0 for the noise standard deviation')
        tmp_dir = tempfile.mkdtemp()
        output = fit_model('s0', self._volume_info, self._protocol, self._mask, tmp_dir, noise_std=None)
        shutil.rmtree(tmp_dir)
        self._logger.info('Done fitting S0 for the noise standard deviation')
        return restore_volumes(output['S0.s0'], self._mask)

