import six
import nibabel as nib
import numpy as np
import numbers

__author__ = 'Robbert Harms'
__date__ = "2015-08-25"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def autodetect_static_maps_loader(data_source):
    """A function to get a static maps loader using the given data source.

    This tries to do auto detecting for the following data sources:
        - StaticMapLoader
        - strings (filenames)
        - ndarray (3d containing the mask)
        - single float

    Args:
        data_source: the data source from which to get a static map loader

    Returns:
        StaticMapLoader: a static map loader instance.
    """
    if isinstance(data_source, StaticMapLoader):
        return data_source
    elif isinstance(data_source, six.string_types):
        return StaticMapFromFileLoader(data_source)
    elif isinstance(data_source, np.ndarray):
        return StaticMapFromArray(data_source)
    elif isinstance(data_source, numbers.Number):
        return StaticMapSingleValue(data_source)

    raise ValueError('The given data source could not be recognized.')


class StaticMapLoader(object):
    """Interface for loading static maps from different sources."""

    def get_data(self, mask):
        """The public method for getting the value of the map.

        Args:
            the mask in use, we need this to convert static maps that are not a single value to one or two
            dimensional arrays.

        Returns:
            ndarray: 3d ndarray containing the static map
        """


class StaticMapFromFileLoader(StaticMapLoader):

    def __init__(self, filename):
        """Loads a static map from the given filename.

        Args:
            filename (str): the filename to load the data from
        """
        self._filename = filename
        self._loaded_data = None

    def get_data(self, mask):
        if self._loaded_data is None:
            self._loaded_data = nib.load(self._filename).get_data()

        from mdt.utils import create_roi
        return create_roi(self._loaded_data, mask)


class StaticMapFromArray(StaticMapLoader):

    def __init__(self, static_map):
        """Adapter for converting a 3d or 4d static map to the right ROI.

        Args:
            ndarray (ndarray): the map data (3d or 4d matrix)
        """
        self._static_map = static_map

    def get_data(self, mask):
        from mdt.utils import create_roi
        return create_roi(self._static_map, mask)


class StaticMapSingleValue(StaticMapLoader):

    def __init__(self, map_value):
        """Adapter for converting a 3d or 4d static map to the right ROI.

        Args:
            map_value (Number): the single map value
        """
        self._map_value = map_value

    def get_data(self, mask):
        return float(self._map_value)
