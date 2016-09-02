import numpy as np
import yaml
import mdt
from mdt.visualization.layouts import GridLayout, Rectangular
import mdt.visualization.layouts

__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MapPlotConfig(object):

    def __init__(self, dimension=2, slice_index=0, volume_index=0, rotate=90, colormap='hot', maps_to_show=(),
                 font_size=14, grid_layout=None, colorbar_nmr_ticks=10, show_axis=True, zoom=None,
                 map_plot_options=None):
        """Container for all plot related settings."""
        self.dimension = dimension
        self.slice_index = slice_index
        self.volume_index = volume_index
        self.rotate = rotate
        self.colormap = colormap
        self.maps_to_show = maps_to_show
        self.font_size = font_size
        self.colorbar_nmr_ticks = colorbar_nmr_ticks
        self.show_axis = show_axis
        self.zoom = zoom or {'x_0': 0, 'y_0': 0, 'x_1': 0, 'y_1': 0}
        self.map_plot_options = map_plot_options or {}
        self.grid_layout = grid_layout or Rectangular()

    @classmethod
    def from_dict(cls, config_dict):
        if 'map_plot_options' not in config_dict:
            config_dict['map_plot_options'] = {}

        for key, value in config_dict['map_plot_options'].items():
            if not isinstance(value, SingleMapConfig):
                config_dict['map_plot_options'][key] = SingleMapConfig.from_dict(value)

        if 'grid_layout' not in config_dict:
            config_dict['grid_layout'] = Rectangular()
        elif not isinstance(config_dict['grid_layout'], GridLayout):
            class_type = getattr(mdt.visualization.layouts, config_dict['grid_layout'][0])
            config_dict['grid_layout'] = class_type(**config_dict['grid_layout'][1])

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, text):
        return cls.from_dict(yaml.load(text))

    def __iter__(self):
        for key, value in self.__dict__.items():
            if key == 'map_plot_options':
                yield key, {k: dict(v) for k, v in value.items()}
            elif key == 'grid_layout':
                yield key, (type(value).__name__, dict(value))
            else:
                yield key, value

    def to_yaml(self):
        return yaml.safe_dump(dict(self))

    def __str__(self):
        return str(dict(self))

    def __eq__(self, other):
        if not isinstance(other, MapPlotConfig):
            return NotImplemented
        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class SingleMapConfig(object):

    def __init__(self, title=None, scale=None, clipping=None, colormap=None):
        super(SingleMapConfig, self).__init__()
        self.title = title
        self.scale = scale or {'min': None, 'max': None}
        self.clipping = clipping or {'min': None, 'max': None}
        self.colormap = colormap

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def __iter__(self):
        for key, value in self.__dict__.items():
            yield key, value

    def __str__(self):
        return str(dict(self))

    def __eq__(self, other):
        if not isinstance(other, SingleMapConfig):
            return NotImplemented

        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class ImageTransformer(object):

    def __init__(self, data):
        """Container for the displayed image data. Has functionality to change the image data."""
        self.data = data

    def rotate(self, factor):
        """Apply rotation and return new a new ImageTransformer object.

        Args:
            factor (int): the angle to rotate by, must be a multiple of 90.
        """
        if factor:
            return ImageTransformer(np.rot90(self.data, factor // 90))
        return self

    def clip(self, clipping):
        """Apply clipping and return new a new ImageTransformer object.

        This function applies basic checks on the clipping dict before clipping.

        Args:
            clipping (dict): the clipping information. Keys: 'min' and 'max'.
        """
        if clipping:
            clipping_min = clipping.get('min', None)
            if clipping_min is None:
                clipping_min = self.data.min()

            clipping_max = clipping.get('max', None)
            if clipping_max is None:
                clipping_max = self.data.max()

            if clipping_min or clipping_max:
                return ImageTransformer(np.clip(self.data, clipping_min, clipping_max))
        return self

    def zoom(self, zoom):
        """Apply zoom and return new a new ImageTransformer object.

        This function applies basic checks on the zoom dict before zooming.

        Args:
           zoom (dict): the zoom information. Keys: 'x_0', 'x_1', 'y_0', 'y_1'
        """
        if zoom:
            correct = all(map(lambda e: e in zoom and zoom[e] is not None and zoom[e] >= 0,
                              ('x_0', 'x_1', 'y_0', 'y_1'))) \
                      and zoom['x_0'] < self.data.shape[1] and zoom['x_1'] < self.data.shape[1] \
                      and zoom['y_0'] < self.data.shape[0] and zoom['y_1'] < self.data.shape[0] \
                      and zoom['x_0'] < zoom['x_1'] and zoom['y_0'] < zoom['y_1']
            if correct:
                return ImageTransformer(self.data[zoom['y_0']:zoom['y_1'], zoom['x_0']:zoom['x_1']])
        return self


class DataInfo(object):

    def __init__(self, maps, directory=None):
        """A container for basic information about the volume maps we are viewing.

        Args:
            maps (dict): the dictionary with the maps to view
            directory (str): the directory from which the maps where loaded
        """
        self.maps = maps
        self.directory = directory
        self.map_info = {key: SingleMapInfo(key, value) for key, value in self.maps.items()}
        self.sorted_keys = list(sorted(maps.keys()))

    @classmethod
    def from_dir(cls, directory):
        return cls(mdt.load_volume_maps(directory), directory)

    def get_max_dimension(self, map_names=None):
        """Get the maximum dimension index in the maps.

        Args:
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: either, 0, 1, 2 as the maximum dimension index in the maps.
        """
        map_names = map_names or self.maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return max(self.map_info[map_name].max_dimension() for map_name in map_names)

    def get_max_slice_index(self, dimension, map_names=None):
        """Get the maximum slice index in the given map on the given dimension.

        Args:
            dimension (int): the dimension we want the slice index of (maximum 3)
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum slice index over the given maps in the given dimension.
        """
        max_dimension = self.get_max_dimension(map_names)
        if not map_names:
            raise ValueError('No maps to search in.')
        if dimension > max_dimension:
            raise ValueError('Dimension can not exceed {}.'.format(max_dimension))
        return max(self.map_info[map_name].max_slice_index(dimension) for map_name in map_names)

    def get_max_volume_index(self, map_names=None):
        """Get the maximum volume index in the given maps.

        Args:
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum volume index in the given list of maps. Starts from 0.
        """
        map_names = map_names or self.maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return max(self.map_info[map_name].max_volume_index() for map_name in map_names)

    def get_index_first_non_zero_slice(self, dimension, map_names=None):
        """Get the index of the first non zero slice in the maps.

        Args:
            dimension (int): the dimension to search in
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the slice index with the first non zero values.
        """
        map_names = map_names or self.maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        for map_name in map_names:
            index = self.map_info[map_name].get_index_first_non_zero_slice(dimension)
            if index is not None:
                return index
        return 0

    def get_max_x(self, dimension, rotate, map_names=None):
        """Get the maximum x index supported over the images.

        In essence this gets the lowest x index found.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the rotation factor by which we rotate the slices within the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum x-index found.
        """
        map_names = map_names or self.maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return min(self.map_info[map_name].get_max_x(dimension, rotate) for map_name in map_names)

    def get_max_y(self, dimension, rotate, map_names=None):
        """Get the maximum y index supported over the images.

        In essence this gets the lowest y index found.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the rotation factor by which we rotate the slices within the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the maximum y-index found.
        """
        map_names = map_names or self.maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return min(self.map_info[map_name].get_max_y(dimension, rotate) for map_name in map_names)


class SingleMapInfo(object):

    def __init__(self, map_name, value):
        """Holds information about a single map.

        Args:
            map_name (str): the name of the map
            value (ndarray): the value of the map
        """
        self.map_name = map_name
        self.value = value

    def max_dimension(self):
        """Get the maximum dimension index in this map.

        The maximum value returned by this method is 2 and the minimum is 0.

        Returns:
            int: in the range 0, 1, 2
        """
        return min(len(self.value.shape), 3) - 1

    def max_slice_index(self, dimension):
        """Get the maximum slice index on the given dimension.

        Args:
            dimension (int): the dimension we want the slice index of (maximum 3)

        Returns:
            int: the maximum slice index in the given dimension.
        """
        return self.value.shape[dimension] - 1

    def max_volume_index(self):
        """Get the maximum volume index in this map.

        The minimum is 0.

        Returns:
            int: the maximum volume index.
        """
        if len(self.value.shape) > 3:
            return self.value.shape[3] - 1
        return 0

    def get_index_first_non_zero_slice(self, dimension):
        """Get the index of the first non zero slice in this map.

        Args:
            dimension (int): the dimension to search in

        Returns:
            int: the slice index with the first non zero values.
        """
        slice_index = [slice(None)] * (self.max_dimension() + 1)
        for index in range(self.value.shape[dimension]):
            slice_index[dimension] = index
            if np.count_nonzero(self.value[slice_index]) > 0:
                return index
        return 0

    def get_max_x(self, dimension, rotate):
        """Get the maximum x index.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the value by which to rotate the slices in the given dimension

        Returns:
            int: the maximum x index
        """
        shape = list(self.value.shape)[0:3]
        del shape[dimension]
        if rotate // 90 % 2 == 0:
            return max(0, shape[1] - 1)
        return max(0, shape[0] - 1)

    def get_max_y(self, dimension, rotate):
        """Get the maximum y index.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the value by which to rotate the slices in the given dimension

        Returns:
            int: the maximum y index
        """
        shape = list(self.value.shape)[0:3]
        del shape[dimension]
        if rotate // 90 % 2 == 0:
            return max(0, shape[0] - 1)
        return max(0, shape[1] - 1)
