import six
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-08-25"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def autodetect_brain_mask_loader(data_source):
    """A function to get a brain mask loader using the given data source.

    This tries to do auto detecting for the following data sources:

        - :class:`BrainMaskLoader`
        - strings (filenames)
        - ndarray (3d containing the mask)

    Args:
        data_source: the data source from which to get a brain_mask loader

    Returns:
        BrainMaskLoader: a brain_mask loader instance.
    """
    if isinstance(data_source, BrainMaskLoader):
        return data_source
    elif isinstance(data_source, six.string_types):
        return BrainMaskFromFileLoader(data_source)
    elif isinstance(data_source, np.ndarray):
        return BrainMaskFromArray(data_source)
    raise ValueError('The given data source could not be recognized.')


class BrainMaskLoader(object):
    """Interface for loading brain_masks from different sources."""

    def get_data(self):
        """The public method used to get an instance of a brain mask.

        Returns:
            ndarray: 3d ndarray containing the brain mask
        """


class BrainMaskFromFileLoader(BrainMaskLoader):

    def __init__(self, filename):
        """Loads a brain mask from the given filename.

        Args:
            filename (str): the filename to use the brain mask from.
        """
        self._filename = filename
        self._brain_mask = None
        self._header = None

    def get_data(self):
        if self._brain_mask is None:
            from mdt.nifti import load_nifti
            self._brain_mask = load_nifti(self._filename).get_data() > 0
        return self._brain_mask


class BrainMaskFromArray(BrainMaskLoader):

    def __init__(self, mask_data):
        """Adapter for returning an already loaded brain mask.

        Args:
            ndarray (ndarray): the brain mask data (3d matrix)
        """
        self._mask_data = mask_data

    def get_data(self):
        return self._mask_data > 0
