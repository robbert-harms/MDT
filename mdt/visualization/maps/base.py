import warnings
from copy import copy, deepcopy
import numbers
import matplotlib.font_manager
import nibabel
import numpy as np
import yaml

import mdt
import mdt.visualization.layouts
from mdt.lib.nifti import load_nifti, NiftiInfoDecorated
from mdt.visualization.dict_conversion import StringConversion, \
    SimpleClassConversion, IntConversion, SimpleListConversion, BooleanConversion, \
    ConvertDictElements, ConvertDynamicFromModule, FloatConversion, WhiteListConversion, ConvertListElements
from mdt.visualization.layouts import Rectangular
from mdt.visualization.maps.utils import get_shortest_unique_names, find_all_nifti_files

__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ConvertibleConfig:
    """Base class for convertible configuration containers.

    Objects implementing this interface return a conversion specification that describes how the implementing
    class can be converted to and from a simple dictionary containing only primitives and simple
    data structures.
    """

    @classmethod
    def get_conversion_info(cls):
        """Get the conversion specification information for this class.

        Returns:
            mdt.visualization.dict_conversion.ConversionSpecification: the conversion specification
        """
        raise NotImplementedError()


class SimpleConvertibleConfig:
    """Offers an simplified implementation of convertible configs.

    In addition to slightly simplifying the creation of the conversion specification, it also implements
    object equality checking (__eq__ and__ne__) as this is commonly used in the configuration objects,
    and it offers a default representation (__repr__) implementation based on the conversion specification.
    """

    @classmethod
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

    @classmethod
    def _get_attribute_conversions(cls):
        """Simplifies the creation of the conversion info by returning a dictionary of conversion specifications.

        Returns:
            dict: a dictionary with for every attribute of the class a conversion specification (instance of
                :class:`~mdt.visualization.dict_conversion.ConversionSpecification`) describing how to
                transform every attribute of the class.
        """
        raise NotImplementedError()

    def __repr__(self):
        return str(self.get_conversion_info().to_dict(self))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        for key, value in self.__dict__.items():
            if value != getattr(other, key) or (value is not None and getattr(other, key) is None) \
                or (value is None and getattr(other, key) is not None):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class DataInfo:
    """Information about the maps we are viewing."""

    def get_map_names(self):
        """Get a list of the names of all the maps in this data info object.

        Returns:
            list: the list of map names
        """
        raise NotImplementedError()

    def get_map_data(self, map_name):
        """Get the data for the indicated map.

        Args:
            map_name (str): the name of the map we want the data of

        Returns:
            ndarray: the data of the given map
        """
        raise NotImplementedError()

    def get_single_map_info(self, map_name):
        """Get an information object for a single map.

        Args:
            map_name (str): the name of the map we want information about

        Returns:
            SingleMapInfo: information object about that map
        """
        raise NotImplementedError()

    def get_file_path(self, map_name):
        """Get the file name of the given map

        Returns:
            None if the map could not be found on dir, else a string with the file path.
        """
        raise NotImplementedError()

    def get_file_paths(self):
        """Get the file paths to each of the maps.

        If one of the maps does not have a file path, None is returned.

        Returns:
            dict: mapping map names to the file paths for each of the maps in this information container.
        """
        raise NotImplementedError()

    def get_max_dimension(self, map_names=None):
        """Get the minimum of the maximum dimension index over the maps

        Args:
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: either, 0, 1, 2 as the maximum dimension index in the maps.
        """
        raise NotImplementedError()

    def get_max_slice_index(self, dimension, map_names=None):
        """Get the maximum slice index in the given map on the given dimension.

        Args:
            dimension (int): the dimension we want the slice index of (maximum 3)
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the minimum of the maximum slice indices over the given maps in the given dimension.
        """
        raise NotImplementedError()

    def get_max_volume_index(self, map_names=None):
        """Get the maximum volume index in the given maps.

        In contrast to the max dimension and max slice index functions, this gives the maximum over all the
        images. This since handling different volumes is implemented in the viewer.

        Args:
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum volume index in the given list of maps. Starts from 0.
        """
        raise NotImplementedError()

    def get_index_first_non_zero_slice(self, dimension, map_names=None):
        """Get the index of the first non zero slice in the maps.

        Args:
            dimension (int): the dimension to search in
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the slice index with the first non zero values.
        """
        raise NotImplementedError()

    def slice_has_data(self, dimension, slice_index, map_names=None):
        """Check if at least one of the maps has non zero numbers on the given slice.

        Args:
            dimension (int): the dimension to search in
            slice_index (int): the index of the slice in the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            bool: true if at least on of the maps has data in the given slice
        """
        raise NotImplementedError()

    def get_max_x_index(self, dimension, rotate=0, map_names=None):
        """Get the maximum x index supported over the images.

        In essence this gets the lowest x index found.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the rotation factor by which we rotate the slices within the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum x-index found.
        """
        raise NotImplementedError()

    def get_max_y_index(self, dimension, rotate=0, map_names=None):
        """Get the maximum y index supported over the images.

        In essence this gets the lowest y index found.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the rotation factor by which we rotate the slices within the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum y-index found.
        """
        raise NotImplementedError()

    def get_bounding_box(self, dimension, slice_index, volume_index, rotate, map_names=None):
        """Get the bounding box of the images.

        Args:
            dimension (int): the dimension to search in
            slice_index (int): the slice index in that dimension
            volume_index (int): the current volume index
            rotate (int): the angle by which to rotate the image before getting the bounding box
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            tuple of :class:`Point2d`: two points designating the upper left corner and the lower right corner of the
                bounding box.
        """
        raise NotImplementedError()


class SimpleDataInfo(DataInfo):

    def __init__(self, maps):
        """A container for basic information about the volume maps we are viewing.

        Args:
            maps (dict): the dictionary with the maps to view, these maps can either be arrays with values,
                nibabel proxy images or SingleMapInfo objects.
        """
        self._input_maps = maps

    @classmethod
    def from_paths(cls, paths):
        """Load all the nifti files from the given paths.

        For paths that are directories we load all the elements inside that directory (but without recursion).

        Args:
            list of str: the list of paths to load (directories and files)

        Returns:
            SimpleDataInfo: the simple data info
        """
        nifti_files = find_all_nifti_files(paths)
        return cls(_load_data_info(nifti_files))

    def get_updated(self, updates=None, removals=None):
        """Get a new simple data info object that includes the given new maps.

        In the case of double map names the old maps are overwritten.

        Args:
            updates (dict): the dictionary with the maps to view, these maps can either be arrays with values,
                nibabel proxy images or SingleMapInfo objects.
            removals (list): a list of maps to remove

        Returns:
            SimpleDataInfo: a new updated data info object
        """
        new_maps = copy(self._input_maps)

        if updates:
            new_maps.update(updates)

        if removals:
            for map_name in removals:
                if map_name in new_maps:
                    del new_maps[map_name]

        return SimpleDataInfo(new_maps)

    def get_map_names(self):
        return list(self._input_maps.keys())

    def get_map_data(self, map_name):
        return self.get_single_map_info(map_name).data

    def get_single_map_info(self, map_name):
        value = self._input_maps[map_name]
        if not isinstance(value, SingleMapInfo):
            return SingleMapInfo(value)
        return value

    def get_file_path(self, map_name):
        return self.get_single_map_info(map_name).file_path

    def get_file_paths(self):
        return {map_name: self.get_file_path(map_name) for map_name in self._input_maps.keys()}

    def get_max_dimension(self, map_names=None):
        """Get the minimum of the maximum dimension index over the maps

        Args:
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: either, 0, 1, 2 as the maximum dimension index in the maps.
        """
        map_names = map_names or self._input_maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return min(self.get_single_map_info(map_name).max_dimension() for map_name in map_names)

    def get_max_slice_index(self, dimension, map_names=None):
        """Get the maximum slice index in the given map on the given dimension.

        Args:
            dimension (int): the dimension we want the slice index of (maximum 3)
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the minimum of the maximum slice indices over the given maps in the given dimension.
        """
        map_names = map_names or self._input_maps.keys()
        max_dimension = self.get_max_dimension(map_names)
        if not map_names:
            raise ValueError('No maps to search in.')
        if dimension > max_dimension:
            raise ValueError('Dimension can not exceed {}.'.format(max_dimension))
        return min(self.get_single_map_info(map_name).max_slice_index(dimension) for map_name in map_names)

    def get_max_volume_index(self, map_names=None):
        """Get the maximum volume index in the given maps.

        In contrast to the max dimension and max slice index functions, this gives the maximum over all the
        images. This since handling different volumes is implemented in the viewer.

        Args:
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum volume index in the given list of maps. Starts from 0.
        """
        map_names = map_names or self._input_maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return max(self.get_single_map_info(map_name).max_volume_index() for map_name in map_names)

    def get_index_first_non_zero_slice(self, dimension, map_names=None):
        """Get the index of the first non zero slice in the maps.

        Args:
            dimension (int): the dimension to search in
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the slice index with the first non zero values.
        """
        map_names = map_names or self._input_maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        for map_name in map_names:
            index = self.get_single_map_info(map_name).get_index_first_non_zero_slice(dimension)
            if index is not None:
                return index
        return 0

    def slice_has_data(self, dimension, slice_index, map_names=None):
        """Check if at least one of the maps has non zero numbers on the given slice.

        Args:
            dimension (int): the dimension to search in
            slice_index (int): the index of the slice in the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            bool: true if at least on of the maps has data in the given slice
        """
        map_names = map_names or self._input_maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        for map_name in map_names:
            if self.get_single_map_info(map_name).slice_has_data(dimension, slice_index):
                return True
        return False

    def get_max_x_index(self, dimension, rotate=0, map_names=None):
        """Get the maximum x index supported over the images.

        In essence this gets the lowest x index found.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the rotation factor by which we rotate the slices within the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum x-index found.
        """
        map_names = map_names or self._input_maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return min(self.get_single_map_info(map_name).get_max_x_index(dimension, rotate) for map_name in map_names)

    def get_max_y_index(self, dimension, rotate=0, map_names=None):
        """Get the maximum y index supported over the images.

        In essence this gets the lowest y index found.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the rotation factor by which we rotate the slices within the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum y-index found.
        """
        map_names = map_names or self._input_maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return min(self.get_single_map_info(map_name).get_max_y_index(dimension, rotate) for map_name in map_names)

    def get_bounding_box(self, dimension, slice_index, volume_index, rotate, map_names=None):
        """Get the bounding box of the images.

        Args:
            dimension (int): the dimension to search in
            slice_index (int): the slice index in that dimension
            volume_index (int): the current volume index
            rotate (int): the angle by which to rotate the image before getting the bounding box
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            tuple of Point: two point designating first the upper left corner and second the lower right corner of the
                bounding box.
        """
        map_names = map_names or self._input_maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        bounding_boxes = [self.get_single_map_info(map_name).get_bounding_box(dimension, slice_index, volume_index, rotate)
                          for map_name in map_names]

        p0x = min([bbox[0].x for bbox in bounding_boxes])
        p0y = min([bbox[0].y for bbox in bounding_boxes])
        p1x = max([bbox[1].x for bbox in bounding_boxes])
        p1y = max([bbox[1].y for bbox in bounding_boxes])

        return Point2d(p0x, p0y), Point2d(p1x, p1y)


class SingleMapInfo:

    def __init__(self, data, file_path=None):
        """Holds information about a single map.

        Args:
            data (ndarray or :class:`nibabel.spatialimages.SpatialImage`): the value of the map or the proxy to it
            file_path (str): optionally, the file path with the location of this map.
                If not set we try to retreive it from the data if the data is of subclass
                :class:`mdt.lib.nifti.NiftiInfoDecorated`.
        """
        self._data = data
        self._shape = self._data.shape
        self._file_path = file_path

        if self._file_path is None:
            if isinstance(data, NiftiInfoDecorated):
                self._file_path = data.nifti_info.filepath

    @classmethod
    def from_file(cls, nifti_path):
        return cls(load_nifti(nifti_path), nifti_path)

    @property
    def shape(self):
        return deepcopy(self._shape)

    @property
    def data(self):
        if isinstance(self._data, nibabel.spatialimages.SpatialImage):
            return self._data.get_data()
        return self._data

    @property
    def file_path(self):
        return self._file_path

    def min(self, mask=None):
        """Get the minimum value in this data.

        If a mask is provided we get the minimum value within the given mask.

        Args:
            mask (ndarray): the mask, we only include elements for which the mask > 0

        Returns:
            float: the minimum value
        """
        if mask is not None:
            return mdt.create_roi(self.data, mask).min()
        return self.data.min()

    def max(self, mask=None):
        """Get the maximum value in this data.

        If a mask is provided we get the maximum value within the given mask.

        Args:
            mask (ndarray): the mask, we only include elements for which the mask > 0

        Returns:
            float: the maximum value
        """
        if mask is not None:
            return mdt.create_roi(self.data, mask).max()
        return self.data.max()

    def min_max(self, mask=None):
        """Get the minimum and maximum value in this data.

        If a mask is provided we get the min and max value within the given mask.

        Infinities and NaN's are ignored by this algorithm.

        Args:
            mask (ndarray): the mask, we only include elements for which the mask > 0

        Returns:
            tuple: (min, max) the minimum and maximum values
        """
        if mask is not None:
            roi = mdt.create_roi(self.data, mask)
            return np.nanmin(roi), np.nanmax(roi)
        return np.nanmin(self.data), np.nanmax(self.data)

    def has_nan(self):
        """Check if this data has any NaNs in it.

        Returns:
            boolean: True if there are NaN's anywhere in the data, false otherwise.
        """
        return np.isnan(self.data).any()

    def max_dimension(self):
        """Get the maximum dimension index in this map.

        The maximum value returned by this method is 2 and the minimum is 0.

        Returns:
            int: in the range 0, 1, 2
        """
        return min(len(self.shape), 3) - 1

    def max_slice_index(self, dimension):
        """Get the maximum slice index on the given dimension.

        Args:
            dimension (int): the dimension we want the slice index of (maximum 3)

        Returns:
            int: the maximum slice index in the given dimension.
        """
        return self.shape[dimension] - 1

    def slice_has_data(self, dimension, slice_index):
        """Check if this map has non zero values in the given slice index.

        Args:
            dimension (int): the dimension we want the slice index of (maximum 3)
            slice_index (int): the slice index to look in

        Returns:
            int: the maximum slice index in the given dimension.
        """
        slice_indexing = [slice(None)] * (self.max_dimension() + 1)
        slice_indexing[dimension] = slice_index
        return np.count_nonzero(self.data[slice_indexing])

    def max_volume_index(self):
        """Get the maximum volume index in this map.

        The minimum is 0.

        Returns:
            int: the maximum volume index.
        """
        if len(self.shape) > 3:
            return self.shape[3] - 1
        return 0

    def get_index_first_non_zero_slice(self, dimension):
        """Get the index of the first non zero slice in this map.

        Args:
            dimension (int): the dimension to search in

        Returns:
            int: the slice index with the first non zero values.
        """
        slice_index = [slice(None)] * (self.max_dimension() + 1)

        if dimension > len(slice_index) - 1:
            raise ValueError('The given dimension {} is not supported.'.format(dimension))

        for index in range(self.shape[dimension]):
            slice_index[dimension] = index
            if np.count_nonzero(self.data[slice_index]) > 0:
                return index
        return 0

    def get_max_x_index(self, dimension, rotate=0):
        """Get the maximum x index.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the value by which to rotate the slices in the given dimension

        Returns:
            int: the maximum x index
        """
        shape = list(self.shape)[0:3]

        if len(shape) > 2:
            del shape[dimension]

        if rotate // 90 % 2 == 0:
            return max(0, shape[1] - 1)
        return max(0, shape[0] - 1)

    def get_max_y_index(self, dimension, rotate=0):
        """Get the maximum y index.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the value by which to rotate the slices in the given dimension

        Returns:
            int: the maximum y index
        """
        shape = list(self.shape)[0:3]

        if len(shape) > 2:
            del shape[dimension]

        if rotate // 90 % 2 == 0:
            return max(0, shape[0] - 1)
        return max(0, shape[1] - 1)

    def get_size_in_dimension(self, dimension, rotate=0):
        """Get the shape of the 2d view on the data in the given dimension.

        This basically returns a pair of (max_x, max_y).

        Args:
            dimension (int): the dimension to search in
            rotate (int): the value by which to rotate the slices in the given dimension

        Returns:
            tuple: (max_x, max_y)
        """
        return self.get_max_x_index(dimension, rotate), self.get_max_y_index(dimension, rotate)

    def get_bounding_box(self, dimension, slice_index, volume_index, rotate):
        """Get the bounding box of this map when displayed using the given indicing.

        This only works if the edges of the images are exactly zero, that is, it only works with masked datasets.

        Args:
            dimension (int): the dimension to search in
            slice_index (int): the slice index in that dimension
            volume_index (int): the current volume index
            rotate (int): the angle by which to rotate the image before getting the bounding box

        Returns:
            tuple of Point: two point designating first the upper left corner and second the lower right corner of the
                bounding box.
        """
        def bbox(image):
            rows = np.any(image, axis=1)
            cols = np.any(image, axis=0)

            rows_where = np.where(rows)

            if np.size(rows_where):
                row_min, row_max = np.where(rows)[0][[0, -1]]
                column_min, column_max = np.where(cols)[0][[0, -1]]

                return row_min, row_max, column_min, column_max
            return 0, image.shape[0]-1, 0, image.shape[1]-1

        if len(self.shape) == 2:
            image = self.data
        else:
            slice_indexing = [slice(None)] * (self.max_dimension() + 1)
            slice_indexing[dimension] = slice_index
            image = self.data[slice_indexing]

        if len(image.shape) > 2:
            if image.shape[2] > 1:
                image = image[..., volume_index]
            else:
                image = image[..., 0]

        if rotate:
            image = np.rot90(image, rotate // 90)

        row_min, row_max, column_min, column_max = bbox(image)
        return Point2d(column_min, row_min), Point2d(column_max, row_max)


class Zoom(SimpleConvertibleConfig):

    def __init__(self, p0, p1):
        """Container for zooming a map between the two given points.

        Args:
            p0 (Point2d): the lower left corner of the zoomed area
            p1 (Point2d): the upper right corner of the zoomed area
        """
        self.p0 = p0
        self.p1 = p1

        if p0.x > p1.x or p0.y > p1.y:
            raise ValueError('The lower left point ({}, {}) should be smaller than the upper right point ({}, {})'.
                             format(p0.x, p0.y, p1.x, p1.y))

        if p0.x < 0 or p0.y < 0 or p1.x < 0 or p1.y < 0:
            raise ValueError('The zoom box ({}, {}), ({}, {}) can not '
                             'be negative in any way.'.format(p0.x, p0.y, p1.x, p1.y))

        if self.p0 is None or self.p1 is None:
            raise ValueError('One of the zoom points is None.')

    @classmethod
    def from_coords(cls, x0, y0, x1, y1):
        return cls(Point2d(x0, y0), Point2d(x1, y1))

    @classmethod
    def no_zoom(cls):
        return cls(Point2d(0, 0), Point2d(0, 0))

    @classmethod
    def _get_attribute_conversions(cls):
        point_converter = Point2d.get_conversion_info()
        return {'p0': point_converter,
                'p1': point_converter}

    def get_rotated(self, rotation, x_dimension, y_dimension):
        """Return a new Zoom instance rotated with the given factor.

        This rotates the zoom box in the same way as the image is rotated.

        Args:
            rotation (int): the rotation by which to rotate in steps of 90 degrees
            x_dimension (int): the dimension of the image in the x coordinate
            y_dimension (int): the dimension of the image in the y coordinate

        Returns:
            Zoom: the rotated instance
        """
        dimensions = [x_dimension, y_dimension]
        p0 = self.p0
        p1 = self.p1

        nmr_90_rotations = rotation % 360 // 90

        for _ in range(nmr_90_rotations):
            dimensions = np.roll(dimensions, 1)

            new_p0 = Point2d(np.min([dimensions[0] - p0.y, dimensions[0] - p1.y]), np.min([p0.x, p1.x]))
            new_p1 = Point2d(np.max([dimensions[0] - p0.y, dimensions[0] - p1.y]), np.max([p0.x, p1.x]))

            p0 = new_p0
            p1 = new_p1

        if p0.x >= dimensions[0] - 1 or p0.x < 0:
            p0 = p0.get_updated(x=0)

        if p0.y >= dimensions[1] - 1 or p0.y < 0:
            p0 = p0.get_updated(y=0)

        if p1.x >= dimensions[0] - 1:
            p1 = p1.get_updated(x=dimensions[0] - 1)

        if p1.y >= dimensions[1] - 1:
            p1 = p1.get_updated(y=dimensions[1] - 1)

        return Zoom(p0, p1)

    def apply(self, data):
        """Apply the zoom to the given 2d array and return the new array.

        Args:
            data (ndarray): the data to zoom in on
        """
        correct = self.p0.x < data.shape[1] and self.p1.x < data.shape[1] \
                  and self.p0.y < data.shape[0] and self.p1.y < data.shape[0] \
                  and self.p0.x < self.p1.x and self.p0.y < self.p1.y
        if correct:
            return data[self.p0.y:self.p1.y, self.p0.x:self.p1.x]
        return data


class ColorbarSettings(SimpleConvertibleConfig):

    def __init__(self, visible=None, nmr_ticks=None, location=None, power_limits=None, round_precision=None):
        """Container for all colorbar related settings.

        Args:
            visible (boolean): if the colorbar is to be shown
            nmr_ticks (int): the number of ticks
            location (str): the location of the colorbar, one of 'right', 'left', 'top' or 'bottom'
            power_limits (tuple): size thresholds for scientific notation. The default is (-3, 4) which uses scientific
                notation for numbers less than 1e-3 or greater than 1e4.
            round_precision (int): how much digits (precision) after the decimal point.
        """
        self.nmr_ticks = nmr_ticks
        self.location = location
        self.visible = visible
        self.power_limits = power_limits
        self.round_precision = round_precision

        if self.location is not None and self.location not in ['left', 'right', 'bottom', 'top']:
            raise ValueError("The colorbar location is '{}' which is not "
                             "one of 'left', 'bottom', 'right', 'top'.".format(str(self.location)))

        if self.nmr_ticks is not None:
            self.nmr_ticks = int(self.nmr_ticks)
            if self.nmr_ticks <= 0:
                raise ValueError("The number of ticks in the colorbar needs to be a positive integer.")

        if self.visible is not None:
            self.visible = bool(self.visible)

        if self.power_limits is not None:
            if not isinstance(self.power_limits, (tuple, list)):
                raise ValueError('The power limits should be a tuple or list.')
            if len(self.power_limits) != 2:
                raise ValueError('The power limits should hold '
                                 'exactly two elements, {} given.'.format(len(self.power_limits)))
            if not all(isinstance(el, numbers.Integral) for el in self.power_limits):
                raise ValueError('The power limits should be integers.')
            if self.power_limits[0] > self.power_limits[1]:
                raise ValueError('The lower power limit should be lower than the upper power limit.')

        if self.round_precision is not None:
            self.round_precision = int(self.round_precision)

    @staticmethod
    def get_default():
        return ColorbarSettings(visible=True, nmr_ticks=4, location='right', power_limits=(-3, 4),
                                round_precision=3)

    @classmethod
    def _get_attribute_conversions(cls):
        return {
            'nmr_ticks': IntConversion(),
            'location': StringConversion(),
            'visible': BooleanConversion(),
            'power_limits': SimpleListConversion(),
            'round_precision': IntConversion()
        }

    def get_preferred(self, attr, other_settings=None):
        """Get the preferred value for the requested attribute.

        Other settings is a list of other colorbar settings object that are asked for the preferred value (in turn),
        if the value of this object is None.

        As a fallback, this will always use the default colorbar settings ``ColorbarSettings.get_default()`` as a final
        default.

        Args:
            attr (str): the attribute requested
            other_settings (list of ColorbarSettings): other settings to try.

        Returns:
            object: the value of the requested object
        """
        if getattr(self, attr) is not None:
            return getattr(self, attr)
        if not other_settings:
            return getattr(self.get_default(), attr)
        for other_setting in other_settings:
            return other_setting.get_preferred(attr, other_settings[1:])


class VoxelAnnotation(SimpleConvertibleConfig):

    valid_text_locations = ['upper left', 'top left', 'upper right',
                            'top right', 'bottom right', 'lower right',
                            'bottom left', 'lower left', 'north', 'south',
                            'east', 'west', 'top', 'bottom', 'left', 'right']

    def __init__(self, voxel_index, font_size=None, text_template=None, marker_size=1, text_location='upper left',
                 text_distance=0.05, arrow_width=1):
        """Container for all voxel highlighting settings.

        Args:
            voxel_index (tuple): a tuple with the voxel index location
            font_size (int): the size of the annotation text
            text_template (str): the text template, can use the placeholders ``{voxel_index}`` and ``{value}``.
            marker_size (float): the rectangular size of the voxel marker
            text_location (str): the location of the text. Valid items are:
                ``upper left``, ``top left``, ``upper right``, ``top right``, ``bottom right``, ``lower right``,
                ``bottom left``, ``lower left``, ``north``, ``south``, ``east``, ``west``, ``top``, ``bottom``, ``left``
                , ``right``.
            text_distance (float): the distance of the textbox to the marker in relative coordinates (0, 1).
            arrow_width (float): the width of the arrow
        """
        self.voxel_index = voxel_index
        self.font_size = font_size
        self.text_template = text_template or "{voxel_index}\n{value:.3g}"
        self.marker_size = marker_size
        self.text_location = text_location
        self.text_distance = text_distance
        self.arrow_width = arrow_width

        if text_location not in self.valid_text_locations:
            raise ValueError('The given text location "{}" is not '
                             'in the list of valid text locations: {}'.format(self.text_location,
                                                                              self.valid_text_locations))

        if not isinstance(self.marker_size, numbers.Real):
            raise ValueError('The marker size should be a real number, {} given.'.format(self.marker_size))

        if not isinstance(self.text_distance, numbers.Real):
            raise ValueError('The text distance should be a real number, {} given.'.format(self.text_distance))

        if not isinstance(self.arrow_width, numbers.Real):
            raise ValueError('The arrow width should be a real number, {} given.'.format(self.arrow_width))

        self.text_template.format(voxel_index=(0, 0, 0), value=0)  # validates the template

    @classmethod
    def _get_attribute_conversions(cls):
        return {
            'voxel_index': SimpleListConversion(),
            'font_size': IntConversion(),
            'text_template': StringConversion(allow_null=False),
            'marker_size': FloatConversion(),
            'text_location': StringConversion(allow_null=False),
            'text_distance': FloatConversion(allow_null=False),
            'arrow_width': FloatConversion(allow_null=False)
        }

    def validate(self, data_info):
        self.text_template.format(voxel_index=(0, 0, 0), value=0)

        if len(self.voxel_index) != 3:
            raise ValueError('The location of the annotation should consist of (x, y, z) '
                             'coordinates, {} given.'.format(self.voxel_index))
        for dim, pos in enumerate(self.voxel_index):
            try:
                max_pos = data_info.get_max_slice_index(dimension=dim)
            except ValueError:
                max_pos = None

            if max_pos is not None:
                if pos > max_pos:
                    raise ValueError('The location of a highlighted voxel in '
                                     'the {} dimension is larger than {}, {} given.'.format(dim, max_pos, pos))

            if pos < 0:
                raise ValueError('The location of a highlighted voxel in '
                                 'the {} dimension is negative, {} given.'.format(dim, pos))


class Point2d(SimpleConvertibleConfig):

    def __init__(self, x, y):
        """Container for a single point"""
        self.x = x
        self.y = y

    def get_updated(self, **kwargs):
        """Get a new :class:`Point2d` with updated arguments.

        Args:
            **kwargs (dict): the new keyword values, when given these take precedence over the current ones.

        Returns:
            Point2d: a new scale with updated values.
        """
        new_values = dict(x=self.x, y=self.y)
        new_values.update(**kwargs)
        return Point2d(**new_values)

    @classmethod
    def _get_attribute_conversions(cls):
        return {'x': IntConversion(allow_null=False),
                'y': IntConversion(allow_null=False)}

    def rotate90(self, nmr_rotations):
        """Rotate this point around a 90 degree angle

        Args:
            nmr_rotations (int): the number of 90 degree rotations, can be negative

        Returns:
            Point2d: the rotated point
        """

        def rotate_coordinate(x, y, nmr_rotations):
            rotation_matrix = np.array([[0, -1],
                                        [1, 0]])
            rx, ry = x, y
            for rotation in range(1, nmr_rotations + 1):
                rx, ry = rotation_matrix.dot([rx, ry])
            return rx, ry

        return Point2d(*rotate_coordinate(self.x, self.y, nmr_rotations))


class Clipping(SimpleConvertibleConfig):

    def __init__(self, vmin=0, vmax=0, use_min=False, use_max=False):
        """Container for the map clipping information"""
        self.vmin = vmin
        self.vmax = vmax
        self.use_min = use_min
        self.use_max = use_max

        if use_min and use_max and vmin > vmax:
            raise ValueError('The minimum clipping ({}) can not be larger than the maximum clipping({})'.format(
                vmin, vmax))

    def apply(self, data):
        """Apply the clipping to the given 2d array and return the new array.

        Args:
           data (ndarray): the data to clip
        """
        if self.use_max or self.use_min:
            clipping_min = data.min()
            if self.use_min:
                clipping_min = self.vmin

            clipping_max = data.max()
            if self.use_max:
                clipping_max = self.vmax

            return np.clip(data, clipping_min, clipping_max)

        return data

    def visible_changes(self, old_clipping):
        """Checks if there are any visible changes between this clipping and the other.

        This method can implement knowledge that allows the visualization routine to check if it
        would need to update the plot or not.

        It expects that the clipping you wish to use is the one on which this method is called.

        Args:
            old_clipping (Clipping): the previous clipping

        Returns:
            bool: if the differences between this clipping and the other would result in visible differences.
        """
        if self.use_min != old_clipping.use_min or self.use_max != old_clipping.use_max:
            return True

        def visible_changes_in_min():
            if self.vmin == old_clipping.vmin:
                return False
            else:
                return self.use_min

        def visible_changes_in_max():
            if self.vmax == old_clipping.vmax:
                return False
            else:
                return self.use_max

        return visible_changes_in_max() or visible_changes_in_min()

    def get_updated(self, **kwargs):
        """Get a new Clipping object with updated arguments.

        Args:
            **kwargs (dict): the new keyword values, when given these take precedence over the current ones.

        Returns:
            Clipping: a new scale with updated values.
        """
        new_values = dict(vmin=self.vmin, vmax=self.vmax, use_min=self.use_min, use_max=self.use_max)
        new_values.update(**kwargs)
        return Clipping(**new_values)

    @classmethod
    def _get_attribute_conversions(cls):
        return {'vmax': FloatConversion(allow_null=False),
                'vmin': FloatConversion(allow_null=False),
                'use_min': BooleanConversion(allow_null=False),
                'use_max': BooleanConversion(allow_null=False)}


class Scale(SimpleConvertibleConfig):

    def __init__(self, vmin=0, vmax=0, use_min=False, use_max=False):
        """Container the map scaling information"""
        self.vmin = vmin
        self.vmax = vmax
        self.use_min = use_min
        self.use_max = use_max

        if use_min and use_max and vmin > vmax:
            raise ValueError('The minimum scale ({}) can not be larger than the maximum scale ({})'.format(vmin, vmax))

    @classmethod
    def _get_attribute_conversions(cls):
        return {'vmax': FloatConversion(allow_null=False),
                'vmin': FloatConversion(allow_null=False),
                'use_min': BooleanConversion(allow_null=False),
                'use_max': BooleanConversion(allow_null=False)}

    def visible_changes(self, old_scale):
        """Checks if there are any visible changes between this scale and the other.

        This method can implement knowledge that allows the visualization routine to check if it
        would need to update the plot or not.

        It expects that the scale you wish to use is the one on which this method is called.

        Args:
            old_scale (Scale): the previous scale

        Returns:
            bool: if the differences between this scale and the other would result in visible differences.
        """
        if self.use_min != old_scale.use_min or self.use_max != old_scale.use_max:
            return True

        def visible_changes_in_min():
            if self.vmin == old_scale.vmin:
                return False
            else:
                return self.use_min

        def visible_changes_in_max():
            if self.vmax == old_scale.vmax:
                return False
            else:
                return self.use_max

        return visible_changes_in_max() or visible_changes_in_min()

    def get_updated(self, **kwargs):
        """Get a new Scale object with updated arguments.

        Args:
            **kwargs (dict): the new keyword values, when given these take precedence over the current ones.

        Returns:
            Scale: a new scale with updated values.
        """
        new_values = dict(vmin=self.vmin, vmax=self.vmax, use_min=self.use_min, use_max=self.use_max)
        new_values.update(**kwargs)
        return Scale(**new_values)


class Font(SimpleConvertibleConfig):

    def __init__(self, family='sans-serif', size=14):
        """Information about the font to use

        Args:
            family: the name of the font to use
            size: the size of the font (> 0).
        """
        self.family = family
        self.size = size

        if family not in self.font_names():
            raise ValueError("The given font \"{}\" is not recognized.".format(family))
        if size < 1:
            raise ValueError("The size ({}) can not be smaller than 1".format(str(size)))

    def get_updated(self, **kwargs):
        """Get a new Font object with updated arguments.

        Args:
            **kwargs (dict): the new keyword values, when given these take precedence over the current ones.

        Returns:
            Font: a new Font with updated values.
        """
        new_values = dict(family=self.family, size=self.size)
        new_values.update(**kwargs)
        return Font(**new_values)

    @property
    def name(self):
        return self.family

    @classmethod
    def font_names(cls):
        """Get the name of supported fonts

        Returns:
            list of str: the name of the supported fonts and font families.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fonts = matplotlib.font_manager.get_fontconfig_fonts()
            names = []

            for font_name in fonts:
                try:
                    names.append(matplotlib.font_manager.FontProperties(fname=font_name).get_name())
                except RuntimeError:
                    pass

        return list(sorted(['sans-serif', 'serif', 'cursive', 'fantasy', 'monospace'])) + list(sorted(names))

    @classmethod
    def _get_attribute_conversions(cls):
        return {'family': StringConversion(),
                'size': IntConversion()}


class MapPlotConfig(SimpleConvertibleConfig):

    def __init__(self, dimension=2, slice_index=0, volume_index=0, rotate=90, colormap='hot', maps_to_show=None,
                 font=None, grid_layout=None, show_axis=False, show_titles=True, zoom=None,
                 map_plot_options=None, interpolation='bilinear', flipud=None, title=None,
                 title_spacing=None, mask_name=None, colorbar_settings=None, annotations=None):
        """Container for all plot related settings.

        Args:
            dimension (int): the dimension we are viewing
            slice_index (int): the slice in the dimension we are viewing
            volume_index (int): in the case of multiple volumes (4th dimension) which index we are in.
            rotate (int): the rotation factor, multiple of 90. By default we rotate 90 degrees to
                show most in-vivo datasets in a natural way.
            colormap (str): the name of the colormap to use
            maps_to_show (list of str): the names of the maps to show
            font (int): the font settings
            grid_layout (GridLayout): the layout of the grid
            show_axis (bool): if we show the axis or not
            show_titles (boolean): the global setting for enabling/disabling the plot titles
            zoom (Zoom): the zoom setting for all the plots
            map_plot_options (dict of SingleMapConfig): per map the map specific plot options
            interpolation (str): one of the available interpolations
            flipud (boolean): if True we flip the image upside down
            title (str): the title to this plot
            title_spacing (float): the spacing between the top of the plots and the title
            mask_name (str): the name of the mask to apply to the maps prior to display
            annotations (list of VoxelAnnotation): the voxel annotations
            colorbar_settings (ColorbarSettings): all colorbar related settings
        """
        super().__init__()

        default_values = self.get_default_values()

        self.dimension = dimension
        self.slice_index = slice_index
        self.volume_index = volume_index
        self.rotate = rotate
        self.colormap = colormap
        self.maps_to_show = maps_to_show or default_values['maps_to_show']
        self.zoom = zoom or default_values['zoom']
        self.font = font or default_values['font']
        self.show_axis = bool(show_axis)
        self.show_titles = bool(show_titles)
        self.grid_layout = grid_layout or default_values['grid_layout']
        self.interpolation = interpolation or default_values['bilinear']
        self.flipud = flipud
        if self.flipud is None:
            self.flipud = default_values['flipud']
        self.map_plot_options = map_plot_options or {}
        self.title = title
        self.title_spacing = title_spacing
        self.mask_name = mask_name
        self.annotations = annotations or []
        self.colorbar_settings = colorbar_settings or default_values['colorbar_settings']

        if interpolation not in self.get_available_interpolations():
            raise ValueError('The given interpolation ({}) is not supported.'.format(interpolation))

        if self.colormap not in self.get_available_colormaps():
            raise ValueError('The given colormap ({}) is not supported.'.format(self.colormap))

        if self.rotate not in [0, 90, 180, 270]:
            raise ValueError('The given rotation ({}) is not supported, use 90 '
                             'degree angles within 360.'.format(self.rotate))

        if self.dimension is None:
            raise ValueError('The dimension can not be None.')

        if self.slice_index is None:
            raise ValueError('The slice index can not be None.')

        if self.volume_index is None:
            raise ValueError('The volume index can not be None.')

        if self.rotate is None:
            raise ValueError('The rotation can not be None.')

        if self.dimension < 0:
            raise ValueError('The dimension can not be smaller than 0, {} given.'.format(self.dimension))

    @classmethod
    def get_available_interpolations(cls):
        return _get_available_interpolations()

    @classmethod
    def get_available_colormaps(cls):
        return _get_available_colormaps()

    @staticmethod
    def get_default_values():
        return {
            'dimension': 2,
            'slice_index': 0,
            'volume_index': 0,
            'rotate': 90,
            'colormap': 'hot',
            'maps_to_show': '',
            'zoom': Zoom.no_zoom(),
            'font': Font(),
            'show_axis': False,
            'show_titles': True,
            'grid_layout': Rectangular(),
            'interpolation': 'bilinear',
            'flipud': False,
            'map_plot_options': {},
            'title': None,
            'title_spacing': None,
            'mask_name': None,
            'annotations': [],
            'colorbar_settings': ColorbarSettings.get_default()
        }

    @classmethod
    def _get_attribute_conversions(cls):
        return {'dimension': IntConversion(),
                'slice_index': IntConversion(),
                'volume_index': IntConversion(),
                'rotate': IntConversion(),
                'colormap': StringConversion(),
                'maps_to_show': SimpleListConversion(),
                'zoom': Zoom.get_conversion_info(),
                'font': Font.get_conversion_info(),
                'show_axis': BooleanConversion(),
                'show_titles': BooleanConversion(),
                'map_plot_options': ConvertDictElements(SingleMapConfig.get_conversion_info()),
                'grid_layout': ConvertDynamicFromModule(mdt.visualization.layouts),
                'interpolation': WhiteListConversion(cls.get_available_interpolations(), 'bilinear'),
                'flipud': BooleanConversion(allow_null=False),
                'title': StringConversion(),
                'title_spacing': FloatConversion(),
                'mask_name': StringConversion(),
                'annotations': ConvertListElements(VoxelAnnotation.get_conversion_info()),
                'colorbar_settings': ColorbarSettings.get_conversion_info()
                }

    @classmethod
    def from_yaml(cls, text):
        return cls.get_conversion_info().from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, config_dict):
        return cls.get_conversion_info().from_dict(config_dict)

    def to_dict(self, non_default_only=False):
        """Export this configuration to a dictionary

        Args:
            non_default_only (boolean): if True, we will only export those options that are not set to their default.

        Returns:
            dict: dict representation of the data
        """
        data = self.get_conversion_info().to_dict(self)
        default_values = self.get_default_values()

        if non_default_only:
            for key, value in list(data['map_plot_options'].items()):
                map_config = SingleMapConfig.get_conversion_info().from_dict(value)

                if map_config == SingleMapConfig():
                    del data['map_plot_options'][key]
                else:
                    data['map_plot_options'][key] = map_config.to_dict(non_default_only=non_default_only)

            for key, default in default_values.items():
                data_value = self._get_attribute_conversions()[key].from_dict(data[key])
                if data_value == default:
                    del data[key]

        return data

    def to_yaml(self, non_default_only=False):
        """Convert this configuration to a YAML string.

        Args:
            non_default_only (boolean): if True, we will only export those options that are not set to their default.

        Returns:
            str: a YAML representation of this configuration.
        """
        return yaml.safe_dump(self.to_dict(non_default_only=non_default_only))

    def visible_changes(self, old_config):
        """Checks if there are any visible changes between this configuration and the other.

        This method can implement knowledge that allows the visualization routine to check if it
        would need to update the plot or not.

        It expects that the configuration you wish to display is the one on which this method is called.

        Args:
            old_config (MapPlotConfig): the previous configuration

        Returns:
            bool: if the differences between this configuration and the other would result in visible differences.
        """
        def visible_difference_in_map_plot_options():
            if not len(self.map_plot_options) and len(old_config.map_plot_options):
                return True

            for key in set(self.map_plot_options.keys()):
                if key in self.maps_to_show:
                    if key not in old_config.map_plot_options:
                        return True
                    if self.map_plot_options[key].visible_changes(old_config.map_plot_options[key]):
                        return True
            return False

        if any(getattr(self, key) != getattr(old_config, key) for key in
               filter(lambda key: key != 'map_plot_options', self.__dict__)):
            return True

        return visible_difference_in_map_plot_options()

    def create_valid(self, data_info):
        """Creates a new configuration object with valid values.

        Args:
            data_info (mdt.visualization.maps.base.DataInfo): the data information

        Returns:
            MapPlotConfig: a valid map plot configuration
        """
        config = deepcopy(self)

        if data_info.get_map_names():
            max_dim = data_info.get_max_dimension()
            if max_dim < config.dimension:
                config.dimension = max_dim

        config.maps_to_show = list(filter(lambda k: k in data_info.get_map_names(), config.maps_to_show))
        if config.mask_name not in data_info.get_map_names():
            config.mask_name = None

        for name in config.maps_to_show:
            if name not in config.map_plot_options:
                config.map_plot_options[name] = SingleMapConfig()

        for name in list(config.map_plot_options):
            if name not in config.maps_to_show:
                del config.map_plot_options[name]

        return config

    def validate(self, data_info):
        """Check if this configuration is valid given the provided data.

        Args:
            data_info (mdt.visualization.maps.base.DataInfo): the data information

        Raises:
            Exception: can raise multiple sorts of exceptions if this config is not valid given the data.
        """
        if data_info.get_map_names():
            self._validate_maps_to_show(data_info)
            self._validate_dimension(data_info)

            for key in self.__dict__:
                if hasattr(self, '_validate_' + key):
                    getattr(self, '_validate_' + key)(data_info)
        return self

    def _validate_maps_to_show(self, data_info):
        if any(map(lambda k: k not in data_info.get_map_names(), self.maps_to_show)):
            raise ValueError('One or more of the given maps to show is not in the data: {}'.
                             format(set(self.maps_to_show).difference(set(data_info.get_map_names()))))

    def _validate_dimension(self, data_info):
        if self.maps_to_show:
            max_dim = data_info.get_max_dimension(map_names=self.maps_to_show)
            if self.dimension is None or self.dimension > max_dim:
                raise ValueError('The dimension ({}) can not be higher than {}.'.format(self.dimension, max_dim))

    def _validate_slice_index(self, data_info):
        if self.maps_to_show:
            max_slice_index = data_info.get_max_slice_index(self.dimension, map_names=self.maps_to_show)
            if self.slice_index is None or self.slice_index > max_slice_index or self.slice_index < 0:
                raise ValueError('The slice index ({}) can not be higher than '
                                 '{} or lower than 0.'.format(self.slice_index, max_slice_index))

    def _validate_volume_index(self, data_info):
        if self.maps_to_show:
            max_volume_index = data_info.get_max_volume_index(map_names=self.maps_to_show)
            if self.volume_index > max_volume_index or self.volume_index < 0:
                raise ValueError('The volume index ({}) can not be higher than '
                                 '{} or lower than 0.'.format(self.volume_index, max_volume_index))

    def _validate_zoom(self, data_info):
        if self.maps_to_show:
            max_x = data_info.get_max_x_index(self.dimension, self.rotate, map_names=self.maps_to_show)
            max_y = data_info.get_max_y_index(self.dimension, self.rotate, map_names=self.maps_to_show)

            if self.zoom.p1.x > max_x:
                raise ValueError('The zoom maximum x ({}) can not be larger than {}'.format(self.zoom.p1.x, max_x))

            if self.zoom.p1.y > max_y:
                raise ValueError('The zoom maximum y ({}) can not be larger than {}'.format(self.zoom.p1.y, max_y))

    def _validate_mask_name(self, data_info):
        if self.mask_name:
            if self.mask_name not in data_info.get_map_names():
                raise ValueError('The given global mask is not found in the list of maps.')

    def _validate_map_plot_options(self, data_info):
        def remove_not_in_data():
            for key in list(self.map_plot_options):
                if key not in data_info.get_map_names():
                    del self.map_plot_options[key]

        def get_validated():
            validated = {}
            for key, value in self.map_plot_options.items():
                if value is not None:
                    validated[key] = value.validate(data_info)
            self.map_plot_options = validated

        def remove_empty():
            empty_config = SingleMapConfig()
            empty_removed = {}
            for key, value in self.map_plot_options.items():
                if value != empty_config:
                    empty_removed[key] = value
            self.map_plot_options = empty_removed

        def create_placeholders():
            for map_name in self.maps_to_show:
                if map_name not in self.map_plot_options:
                    self.map_plot_options[map_name] = SingleMapConfig()

        remove_not_in_data()
        get_validated()
        remove_empty()
        create_placeholders()

    def _validate_annotations(self, data_info):
        if self.annotations is None:
            self.annotations = []

        for annotation in self.annotations:
            annotation.validate(data_info)


class SingleMapConfig(SimpleConvertibleConfig):

    def __init__(self, title=None, scale=None, clipping=None, colormap=None, colorbar_label=None,
                 show_title=None, title_spacing=None, mask_name=None, interpret_as_colormap=False,
                 colormap_weight_map=None,
                 colormap_order=None, colorbar_settings=None):
        """Creates the configuration for a single map plot.

        Args:
            title (str): the title of this plot, can contain latex using the matplotlib latex syntax
            scale (Scale): the scaling for the values in this map
            clipping (Clipping): the clipping to apply to the values prior to plotting
            colormap (str): the matplotlib colormap to use
            colorbar_label (str): the label for the colorbar
            show_title (boolean): if we want to show the title or not
            title_spacing (float): the spacing between the top of the plots and the title
            mask_name (str): the name of the mask used to mask the data prior to visualization
            interpret_as_colormap (boolean): if this is set to True and the referring map is a 4d volume with
                a vector of length 3 on the last dimension, then we can interpret this map as a colormap. This means
                that the elements of the last dimensions are used as (R, G, B) scalar values.
            colormap_weight_map (str): the name of another map to use as a scaling factor for this map. This is only
                used when ``interpret_as_colormap`` is set to True. This scales this map with the specified weight map.
            colormap_order (str): only used if ``interpret_as_colormap`` is used. This defines the order of the RGB
                components of the data. Valid strings are permutations of the letters RGB.
            colorbar_settings (ColorbarSettings): all colorbar related settings
        """
        super().__init__()

        default_values = self.get_default_values()

        self.title = title
        self.title_spacing = title_spacing
        self.scale = scale or default_values['scale']
        self.clipping = clipping or default_values['clipping']
        self.colormap = colormap
        self.colorbar_label = colorbar_label
        self.show_title = bool(show_title) if show_title is not None else default_values['show_title']
        self.mask_name = mask_name
        self.interpret_as_colormap = bool(interpret_as_colormap)
        self.colormap_weight_map = colormap_weight_map
        self.colormap_order = colormap_order
        self.colorbar_settings = colorbar_settings or default_values['colorbar_settings']

        if self.colormap is not None and self.colormap not in self.get_available_colormaps():
            raise ValueError('The given colormap ({}) is not supported.'.format(self.colormap))

        if colormap_order:
            if len(colormap_order) > 3 or not all(color in colormap_order.lower() for color in 'rgb'):
                raise ValueError('Incorrect colormap order specification, '
                                 'only permutations of "rgb" are allowed.'.format(colormap_order))

    @staticmethod
    def get_default_values():
        return {
            'title': None,
            'title_spacing': None,
            'scale': Scale(),
            'clipping': Clipping(),
            'colormap': None,
            'colorbar_label': None,
            'show_title': None,
            'mask_name': None,
            'interpret_as_colormap': False,
            'colormap_weight_map': None,
            'colormap_order': None,
            'colorbar_settings': ColorbarSettings()
        }

    @classmethod
    def _get_attribute_conversions(cls):
        return {'title': StringConversion(),
                'scale': Scale.get_conversion_info(),
                'clipping': Clipping.get_conversion_info(),
                'colormap': StringConversion(),
                'colorbar_label': StringConversion(),
                'title_spacing': FloatConversion(),
                'mask_name': StringConversion(),
                'show_title': BooleanConversion(),
                'interpret_as_colormap': BooleanConversion(allow_null=False),
                'colormap_weight_map': StringConversion(),
                'colormap_order': StringConversion(),
                'colorbar_settings': ColorbarSettings.get_conversion_info()}

    @classmethod
    def get_available_colormaps(cls):
        return _get_available_colormaps()

    @classmethod
    def from_yaml(cls, text):
        return cls.get_conversion_info().from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, config_dict):
        return cls.get_conversion_info().from_dict(config_dict)

    def to_dict(self, non_default_only=False):
        """Export this configuration to a dictionary

        Args:
            non_default_only (boolean): if True, we will only export those options that are not set to their default.

        Returns:
            dict: dict representation of the data
        """
        data = self.get_conversion_info().to_dict(self)
        default_values = self.get_default_values()

        if non_default_only:
            for key, default in default_values.items():
                data_value = self._get_attribute_conversions()[key].from_dict(data[key])
                if data_value == default:
                    del data[key]

        return data

    def to_yaml(self, non_default_only=False):
        """Convert this configuration to a YAML string.

        Args:
            non_default_only (boolean): if True, we will only export those options that are not set to their default.

        Returns:
            str: a YAML representation of this configuration.
        """
        return yaml.safe_dump(self.to_dict(non_default_only=non_default_only))

    def visible_changes(self, old_config):
        """Checks if there are any visible changes between this configuration and the other.

        This method can implement knowledge that allows the visualization routine to check if it
        would need to update the plot or not.

        It expects that the configuration you wish to display is the one on which this method is called.

        Args:
            old_config (SingleMapConfig): the previous configuration

        Returns:
            bool: if the differences between this configuration and the other would result in visible differences.
        """
        def filtered_attributes():
            filtered = ['scale', 'clipping']
            return [key for key in self.__dict__ if key not in filtered]

        def visible_changes_in_scale():
            return self.scale.visible_changes(old_config.scale)

        def visible_changes_in_clipping():
            return self.clipping.visible_changes(old_config.clipping)

        if any(getattr(self, key) != getattr(old_config, key) for key in filtered_attributes()):
            return True

        return visible_changes_in_clipping() or visible_changes_in_scale()

    def validate(self, data_info):
        for key in self.__dict__:
            if hasattr(self, '_validate_' + key):
                getattr(self, '_validate_' + key)(data_info)
        return self

    def _validate_mask_name(self, data_info):
        if self.mask_name is not None and self.mask_name not in data_info.get_map_names():
            raise ValueError('The given mask name "{}" does not exist.'.format(self.mask_name))

    def _validate_colormap_weight_map(self, data_info):
        if self.colormap_weight_map is not None and self.colormap_weight_map not in data_info.get_map_names():
            raise ValueError('The given colormap weight map "{}" does not exist.'.format(self.colormap_weight_map))


def _get_available_interpolations():
    """The available interpolations for either the general map plot config or the map specifics.

    Do not call these for outside use, rather, consult the class method of the specific config you want to change.

    Returns:
        list of str: the list of available interpolations.
    """
    return ['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']


def _get_available_colormaps():
    """The available colormaps for either the general map plot config or the map specifics.

    Do not call these for outside use, rather, consult the class method of the specific config you want to change.

    Returns:
        list of str: the list of available colormaps.
    """
    return sorted(matplotlib.cm.datad)


def _load_data_info(nifti_files):
    """Load the data info for all the nifti files.

    In the case of conflicting simplified map names we name the maps to the shortest unique names.

    Returns:
        dict[str, SingleMapInfo]: the dictionary with the single map information
    """
    data_info = {}
    for ind, map_name in enumerate(get_shortest_unique_names(nifti_files)):
        data_info[map_name] = SingleMapInfo.from_file(nifti_files[ind])
    return data_info
