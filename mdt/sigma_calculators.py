__author__ = 'Robbert Harms'
__date__ = "2015-10-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SigmaCalculator(object):

    def calculate_sigma(self, dwi_info, protocol, mask):
        """Calculate the sigma used in the evaluation models for the multi-compartment models.

        Args:
            dwi_info (string): Either an (ndarray, img_header) tuple or the full path to the volume (4d signal data).
            protocol (Protocol or string): A protocol object with the right protocol for the given data,
                or a string object with a filename to the given file.
            brain_mask (string): A full path to a mask file that can optionally be used. If None given, no mask is used.

        Returns:
            float: single value representing the sigma for the given volume
        """


# class