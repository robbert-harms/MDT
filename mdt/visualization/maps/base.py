import glob
import warnings
import numpy as np
import yaml
import matplotlib.font_manager
import mdt
import mdt.visualization.layouts
from mdt.visualization.dict_conversion import StringConversion, \
    SimpleClassConversion, IntConversion, SimpleListConversion, BooleanConversion, \
    ConvertDictElements, ConvertDynamicFromModule, FloatConversion, WhiteListConversion
from mdt.visualization.layouts import Rectangular

__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MapPlotConfig(object):

    def __init__(self, dimension=2, slice_index=0, volume_index=0, rotate=90, colormap='hot', maps_to_show=None,
                 font=None, grid_layout=None, colorbar_nmr_ticks=10, show_axis=True, zoom=None,
                 map_plot_options=None, interpolation='bilinear', flipud=None):
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
            colorbar_nmr_ticks (int): the number of ticks on the colorbar
            show_axis (bool): if we show the axis or not
            zoom (Zoom): the zoom setting for all the plots
            map_plot_options (dict): per map the map specific plot options
            interpolation (str): one of the available interpolations
            flipud (boolean): if True we flip the image upside down
        """
        super(MapPlotConfig, self).__init__()
        self.dimension = dimension
        self.slice_index = slice_index
        self.volume_index = volume_index
        self.rotate = rotate
        self.colormap = colormap
        self.maps_to_show = maps_to_show or []
        self.zoom = zoom or Zoom.no_zoom()
        self.font = font or Font()
        self.colorbar_nmr_ticks = colorbar_nmr_ticks
        self.show_axis = show_axis
        if self.show_axis is None:
            self.show_axis = True
        self.grid_layout = grid_layout or Rectangular()
        self.interpolation = interpolation or 'bilinear'
        self.flipud = flipud
        if self.flipud is None:
            self.flipud = False
        self.map_plot_options = map_plot_options or {}

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
        return get_available_interpolations()

    @classmethod
    def get_available_colormaps(cls):
        return get_available_colormaps()

    @classmethod
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

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
                'colorbar_nmr_ticks': IntConversion(),
                'show_axis': BooleanConversion(),
                'map_plot_options': ConvertDictElements(SingleMapConfig.get_conversion_info()),
                'grid_layout': ConvertDynamicFromModule(mdt.visualization.layouts),
                'interpolation': WhiteListConversion(cls.get_available_interpolations(), 'bilinear'),
                'flipud': BooleanConversion(allow_null=False),
                }

    @classmethod
    def from_yaml(cls, text):
        return cls.get_conversion_info().from_dict(yaml.load(text))

    @classmethod
    def from_dict(cls, config_dict):
        return cls.get_conversion_info().from_dict(config_dict)

    def to_dict(self):
        return self.get_conversion_info().to_dict(self)

    def to_yaml(self):
        return yaml.safe_dump(self.get_conversion_info().to_dict(self))

    def __repr__(self):
        return str(self.get_conversion_info().to_dict(self))

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

    def __init__(self, title=None, scale=None, clipping=None, colormap=None, colorbar_label=None):
        super(SingleMapConfig, self).__init__()
        self.title = title
        self.scale = scale or Scale()
        self.clipping = clipping or Clipping()
        self.colormap = colormap
        self.colorbar_label = colorbar_label

        if self.colormap is not None and self.colormap not in self.get_available_colormaps():
            raise ValueError('The given colormap ({}) is not supported.'.format(self.colormap))

    @classmethod
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

    @classmethod
    def _get_attribute_conversions(cls):
        return {'title': StringConversion(),
                'scale': Scale.get_conversion_info(),
                'clipping': Clipping.get_conversion_info(),
                'colormap': StringConversion(),
                'colorbar_label': StringConversion()}

    @classmethod
    def get_available_colormaps(cls):
        return get_available_colormaps()

    @classmethod
    def from_yaml(cls, text):
        return cls.get_conversion_info().from_dict(yaml.load(text))

    @classmethod
    def from_dict(cls, config_dict):
        return cls.get_conversion_info().from_dict(config_dict)

    def to_dict(self):
        return self.get_conversion_info().to_dict(self)

    def to_yaml(self):
        return yaml.safe_dump(self.get_conversion_info().to_dict(self))

    def __repr__(self):
        return str(self.get_conversion_info().to_dict(self))

    def __eq__(self, other):
        if not isinstance(other, SingleMapConfig):
            return NotImplemented

        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class Zoom(object):

    def __init__(self, p0, p1):
        """Container for zooming a map between the two given points.

        Args:
            p0 (Point): the lower left corner of the zoomed area
            p1 (Point): the upper right corner of the zoomed area
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
        return cls(Point(x0, y0), Point(x1, y1))

    @classmethod
    def no_zoom(cls):
        return cls(Point(0, 0), Point(0, 0))

    @classmethod
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

    @classmethod
    def _get_attribute_conversions(cls):
        point_converter = Point.get_conversion_info()
        return {'p0': point_converter,
                'p1': point_converter}

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

    def __repr__(self):
        return str(self.get_conversion_info().to_dict(self))

    def __eq__(self, other):
        if not isinstance(other, Zoom):
            return NotImplemented

        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class Point(object):

    def __init__(self, x, y):
        """Container for a single point"""
        self.x = x
        self.y = y

    def get_updated(self, **kwargs):
        """Get a new Point object with updated arguments.

        Args:
            **kwargs (dict): the new keyword values, when given these take precedence over the current ones.

        Returns:
            Point: a new scale with updated values.
        """
        new_values = dict(x=self.x, y=self.y)
        new_values.update(**kwargs)
        return Point(**new_values)

    @classmethod
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

    @classmethod
    def _get_attribute_conversions(cls):
        return {'x': IntConversion(allow_null=False),
                'y': IntConversion(allow_null=False)}

    def rotate90(self, nmr_rotations):
        """Rotate this point around a 90 degree angle

        Args:
            nmr_rotations (int): the number of 90 degreee rotations, can be negative

        Returns:
            Point: the rotated point
        """

        def rotate_coordinate(x, y, nmr_rotations):
            rotation_matrix = np.array([[0, -1],
                                        [1, 0]])
            rx, ry = x, y
            for rotation in range(1, nmr_rotations + 1):
                rx, ry = rotation_matrix.dot([rx, ry])
            return rx, ry

        return Point(*rotate_coordinate(self.x, self.y, nmr_rotations))

    def __repr__(self):
        return 'Point(x={}, y={})'.format(self.x, self.y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented

        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class Clipping(object):

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
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

    @classmethod
    def _get_attribute_conversions(cls):
        return {'vmax': FloatConversion(allow_null=False),
                'vmin': FloatConversion(allow_null=False),
                'use_min': BooleanConversion(allow_null=False),
                'use_max': BooleanConversion(allow_null=False)}

    def __eq__(self, other):
        if not isinstance(other, Clipping):
            return NotImplemented

        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class Scale(object):

    def __init__(self, vmin=0, vmax=0, use_min=False, use_max=False):
        """Container the map scaling information"""
        self.vmin = vmin
        self.vmax = vmax
        self.use_min = use_min
        self.use_max = use_max

        if use_min and use_max and vmin > vmax:
            raise ValueError('The minimum scale ({}) can not be larger than the maximum scale ({})'.format(vmin, vmax))

    @classmethod
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

    @classmethod
    def _get_attribute_conversions(cls):
        return {'vmax': FloatConversion(allow_null=False),
                'vmin': FloatConversion(allow_null=False),
                'use_min': BooleanConversion(allow_null=False),
                'use_max': BooleanConversion(allow_null=False)}

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

    def __eq__(self, other):
        if not isinstance(other, Scale):
            return NotImplemented

        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class Font(object):

    def __init__(self, family='sans-serif', size=14):
        """Information about the font to use

        Args:
            name: the name of the font to use
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
            names = [matplotlib.font_manager.FontProperties(fname=font_name).get_name() for font_name in fonts]
        return list(sorted(['sans-serif', 'serif', 'cursive', 'fantasy', 'monospace'])) + list(sorted(names))

    @classmethod
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

    @classmethod
    def _get_attribute_conversions(cls):
        return {'family': StringConversion(),
                'size': IntConversion()}

    def __eq__(self, other):
        if not isinstance(other, Font):
            return NotImplemented

        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


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
        if directory is None:
            return cls({}, None)
        return cls(mdt.load_volume_maps(directory), directory)

    def get_file_name(self, map_name):
        """Get the file name of the given map

        Returns:
            None if the map could not be found on dir, else a string with the file path.
        """
        if not self.directory:
            return None

        items = list(glob.glob(self.directory + '/{}.nii*'.format(map_name)))
        if items:
            return items[0]

        return None

    def get_max_dimension(self, map_names=None):
        """Get the minimum of the maximum dimension index over the maps

        Args:
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: either, 0, 1, 2 as the maximum dimension index in the maps.
        """
        map_names = map_names or self.maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        return min(self.map_info[map_name].max_dimension() for map_name in map_names)

    def get_max_slice_index(self, dimension, map_names=None):
        """Get the maximum slice index in the given map on the given dimension.

        Args:
            dimension (int): the dimension we want the slice index of (maximum 3)
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            int: the minimum of the maximum slice indices over the given maps in the given dimension.
        """
        map_names = map_names or self.maps.keys()
        max_dimension = self.get_max_dimension(map_names)
        if not map_names:
            raise ValueError('No maps to search in.')
        if dimension > max_dimension:
            raise ValueError('Dimension can not exceed {}.'.format(max_dimension))
        return min(self.map_info[map_name].max_slice_index(dimension) for map_name in map_names)

    def get_max_volume_index(self, map_names=None):
        """Get the maximum volume index in the given maps.

        In contrast to the max dimension and max slice index functions, this gives the maximum over all the
        images. This since handling different volumes is implemented in the viewer.

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

    def slice_has_data(self, dimension, slice_index, map_names=None):
        """Check if at least one of the maps has non zero numbers on the given slice.

        Args:
            dimension (int): the dimension to search in
            slice_index (int): the index of the slice in the given dimension
            map_names (list of str): if given we will only scan the given list of maps

        Returns:
            bool: true if at least on of the maps has data in the given slice
        """
        map_names = map_names or self.maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        for map_name in map_names:
            if self.map_info[map_name].slice_has_data(dimension, slice_index):
                return True
        return False

    def get_max_x(self, dimension, rotate=0, map_names=None):
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

    def get_max_y(self, dimension, rotate=0, map_names=None):
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
        map_names = map_names or self.maps.keys()
        if not map_names:
            raise ValueError('No maps to search in.')
        bounding_boxes = [self.map_info[map_name].get_bounding_box(dimension, slice_index, volume_index, rotate)
                          for map_name in map_names]

        p0x = min([bbox[0].x for bbox in bounding_boxes])
        p0y = min([bbox[0].y for bbox in bounding_boxes])
        p1x = max([bbox[1].x for bbox in bounding_boxes])
        p1y = max([bbox[1].y for bbox in bounding_boxes])

        return Point(p0x, p0y), Point(p1x, p1y)


class SingleMapInfo(object):

    def __init__(self, map_name, data):
        """Holds information about a single map.

        Args:
            map_name (str): the name of the map
            data (ndarray): the value of the map
        """
        self.map_name = map_name
        self.data = data

    def max_dimension(self):
        """Get the maximum dimension index in this map.

        The maximum value returned by this method is 2 and the minimum is 0.

        Returns:
            int: in the range 0, 1, 2
        """
        return min(len(self.data.shape), 3) - 1

    def max_slice_index(self, dimension):
        """Get the maximum slice index on the given dimension.

        Args:
            dimension (int): the dimension we want the slice index of (maximum 3)

        Returns:
            int: the maximum slice index in the given dimension.
        """
        return self.data.shape[dimension] - 1

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
        if len(self.data.shape) > 3:
            return self.data.shape[3] - 1
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

        for index in range(self.data.shape[dimension]):
            slice_index[dimension] = index
            if np.count_nonzero(self.data[slice_index]) > 0:
                return index
        return 0

    def get_max_x(self, dimension, rotate=0):
        """Get the maximum x index.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the value by which to rotate the slices in the given dimension

        Returns:
            int: the maximum x index
        """
        shape = list(self.data.shape)[0:3]
        del shape[dimension]
        if rotate // 90 % 2 == 0:
            return max(0, shape[1] - 1)
        return max(0, shape[0] - 1)

    def get_max_y(self, dimension, rotate=0):
        """Get the maximum y index.

        Args:
            dimension (int): the dimension to search in
            rotate (int): the value by which to rotate the slices in the given dimension

        Returns:
            int: the maximum y index
        """
        shape = list(self.data.shape)[0:3]
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
        return self.get_max_x(dimension, rotate), self.get_max_y(dimension, rotate)

    def get_bounding_box(self, dimension, slice_index, volume_index, rotate):
        """Get the bounding box of this map when displayed using the given indicing.

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
            row_min, row_max = np.where(rows)[0][[0, -1]]
            column_min, column_max = np.where(cols)[0][[0, -1]]

            return row_min, row_max, column_min, column_max

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

        return Point(column_min, row_min), Point(column_max, row_max)


def get_available_interpolations():
    """The available interpolations for either the general map plot config or the map specifics.

    Do not call these for outside use, rather, consult the class method of the specific config you want to change.

    Returns:
        list of str: the list of available interpolations.
    """
    return ['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']


def get_available_colormaps():
    """The available colormaps for either the general map plot config or the map specifics.

    Do not call these for outside use, rather, consult the class method of the specific config you want to change.

    Returns:
        list of str: the list of available colormaps.
    """
    return sorted(matplotlib.cm.datad)
