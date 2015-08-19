import logging
import os
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from six import string_types
from mdt.protocols import load_from_protocol
import nibabel as nib
from mot import runtime_configuration
from mot.cl_routines.filters.median import MedianFilter

__author__ = 'Robbert Harms'
__date__ = "2015-07-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def create_median_otsu_brain_mask(dwi_info, protocol, mask_threshold=0, **kwargs):
    """Create a brain mask using the given volume.

    If output_fname is given this will also write the mask to the given filename.

    Args:
        dwi_info (string or (image, header) pair or image):
            - the filename of the input file;
            - or a tuple with as first index a ndarray with the DWI and as second index the header;
            - or only the image as an ndarray
        protocol (string or Protocol): The filename of the protocol file or a Protocol object
        mask_threshold (double): everything below this threshold is masked away
        **kwargs: the additional arguments for median_otsu.

    Returns:
        ndarray: The created brain mask
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting calculating a brain mask')

    if isinstance(dwi_info, string_types):
        signal_img = nib.load(dwi_info)
        dwi = signal_img.get_data()
    elif isinstance(dwi_info, (tuple, list)):
        dwi = dwi_info[0]
    else:
        dwi = dwi_info

    if isinstance(protocol, string_types):
        protocol = load_from_protocol(protocol)

    if len(dwi.shape) == 4:
        unweighted_ind = protocol.get_unweighted_indices()
        if len(unweighted_ind):
            unweighted = np.mean(dwi[..., unweighted_ind], axis=3)
        else:
            unweighted = np.mean(dwi, axis=3)
    else:
        unweighted = dwi.copy()

    brain_mask = median_otsu(unweighted, **kwargs)
    brain_mask = brain_mask > 0

    if mask_threshold:
        brain_mask = np.mean(dwi[..., protocol.get_weighted_indices()], axis=3) * brain_mask > mask_threshold

    logger.info('Finished calculating a brain mask')

    return brain_mask


def create_write_median_otsu_brain_mask(dwi_info, protocol, output_fname, **kwargs):
    """Write a brain mask using the given volume and output as the given volume.

    Args:
        dwi_info (string or (image, header) pair): the filename of the input file or a tuple with as
            first index a ndarray with the DWI and as second index the header or only the image.
        protocol (string or Protocol): The filename of the protocol file or a Protocol object
        output_fname (string): the filename of the output file (the extracted brain mask)
            If None, no output is written. If dwi_info is only an image also no file is written.

    Returns:
        ndarray: The created brain mask
    """
    if not os.path.isdir(os.path.dirname(output_fname)):
        os.makedirs(os.path.dirname(output_fname))

    if isinstance(dwi_info, string_types):
        signal_img = nib.load(dwi_info)
        dwi = signal_img.get_data()
        header = signal_img.get_header()
    else:
        dwi = dwi_info[0]
        header = dwi_info[1]

    mask = create_median_otsu_brain_mask(dwi, protocol, **kwargs)
    nib.Nifti1Image(mask, None, header).to_filename(output_fname)

    return mask


def median_otsu(unweighted_volume, median_radius=4, numpass=4, dilate=1, cl_environments=None):
    """ Simple brain extraction tool method for images from DWI data

    This function is inspired from the median_otsu function from dipy
    and is copied here to remove a dependency.

    It uses a median filter smoothing of the unweighted_volume
    automatic histogram Otsu thresholding technique, hence the name
    *median_otsu*.

    This function is inspired from Mrtrix's bet which has default values
    ``median_radius=3``, ``numpass=2``. However, from tests on multiple 1.5T
    and 3T data. From GE, Philips, Siemens, the most robust choice is
    ``median_radius=4``, ``numpass=4``.

    Args:
        unweighted_volume (ndarray): ndarray of the unweighted volumes brain volumes
        median_radius (int): Radius (in voxels) of the applied median filter(default 4)
        numpass (int) Number of pass of the median filter (default 4)
        dilate (None or int): optional number of iterations for binary dilation
        cl_environments (None): the CL environments to use for the filtering

    Returns:
        ndarray: a 3D ndarray with the binary brain mask
    """
    b0vol = unweighted_volume

    m = MedianFilter(median_radius,
                     runtime_configuration.runtime_config['cl_environments'],
                     runtime_configuration.runtime_config['load_balancer'])
    m.cl_environments = cl_environments
    for i in range(0, numpass):
        b0vol = m.filter(b0vol)

    thresh = _otsu(b0vol)
    mask = b0vol > thresh

    if dilate is not None:
        cross = generate_binary_structure(3, 1)
        mask = binary_dilation(mask, cross, iterations=dilate)

    return mask


def _otsu(image, nbins=256):
    """
    Return threshold value based on Otsu's method.
    Copied from scikit-image to remove dependency.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    hist, bin_centers = np.histogram(image, nbins)
    hist = hist.astype(np.float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers[1:]) / weight1
    mean2 = (np.cumsum((hist * bin_centers[1:])[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold