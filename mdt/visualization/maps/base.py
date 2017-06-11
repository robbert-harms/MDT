from copy import copy
import os
import inspect
import warnings

import nibabel
import numpy as np
import yaml
import matplotlib.font_manager
import mdt
import mdt.visualization.layouts
from mdt.nifti import yield_nifti_info, load_nifti
from mdt.visualization.dict_conversion import StringConversion, \
    SimpleClassConversion, IntConversion, SimpleListConversion, BooleanConversion, \
    ConvertDictElements, ConvertDynamicFromModule, FloatConversion, WhiteListConversion, SimpleDictConversion, \
    ConvertListElements
from mdt.visualization.layouts import Rectangular


__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ConvertibleConfig(object):
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


class SimpleConvertibleConfig(object):
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
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class MapPlotConfig(SimpleConvertibleConfig):

    def __init__(self, dimension=2, slice_index=0, volume_index=0, rotate=90, colormap='hot', maps_to_show=None,
                 font=None, grid_layout=None, colorbar_nmr_ticks=4, show_axis=False, zoom=None, show_colorbar=True,
                 map_plot_options=None, interpolation='bilinear', flipud=None,
                 title=None, mask_name=None, drawable_patches=None):
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
            show_colorbar (boolean): the global setting for enabling/disabling the colorbar
            map_plot_options (dict): per map the map specific plot options
            interpolation (str): one of the available interpolations
            flipud (boolean): if True we flip the image upside down
            title (str): the title to this plot
            mask_name (str): the name of the mask to apply to the maps prior to display
            drawable_patches (list of DrawablePatch): the patches that need to be drawn on every displayed image
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
        self.show_colorbar = show_colorbar
        self.show_axis = show_axis
        if self.show_axis is None:
            self.show_axis = True
        self.grid_layout = grid_layout or Rectangular()
        self.interpolation = interpolation or 'bilinear'
        self.flipud = flipud
        if self.flipud is None:
            self.flipud = False
        self.map_plot_options = map_plot_options or {}
        self.title = title
        self.mask_name = mask_name
        self.drawable_patches = drawable_patches or []

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
                'title': StringConversion(),
                'mask_name': StringConversion(),
                'show_colorbar': BooleanConversion(),
                'drawable_patches': ConvertListElements(DrawablePatches.get_conversion_info())
                }

    @classmethod
    def from_yaml(cls, text):
        return cls.get_conversion_info().from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, config_dict):
        return cls.get_conversion_info().from_dict(config_dict)

    def to_dict(self):
        return self.get_conversion_info().to_dict(self)

    def to_yaml(self):
        return yaml.safe_dump(self.get_conversion_info().to_dict(self))

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

    def validate(self, data_info):
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
        for key in list(self.map_plot_options):
            if key not in data_info.get_map_names():
                del self.map_plot_options[key]

        for key, value in self.map_plot_options.items():
            if value is not None:
                self.map_plot_options[key] = value.validate(data_info)


class SingleMapConfig(SimpleConvertibleConfig):

    def __init__(self, title=None, scale=None, clipping=None, colormap=None, colorbar_label=None, show_colorbar=True,
                 title_spacing=None, mask_name=None, drawable_patches=None):
        """Creates the configuration for a single map plot.

        Args:
            title (str): the title of this plot, can contain latex using the matplotlib latex syntax
            scale (Scale): the scaling for the values in this map
            clipping (Clipping): the clipping to apply to the values prior to plotting
            colormap (str): the matplotlib colormap to use
            colorbar_label (str): the label for the colorbar
            show_colorbar (boolean): if we want to show the colorbar or not
            title_spacing (float): the spacing between the top of the plots and the title
            mask_name (str): the name of the mask used to mask the data prior to visualization
            drawable_patches (list of DrawablePatch): the patches that need to be drawn on every displayed image
        """
        super(SingleMapConfig, self).__init__()
        self.title = title
        self.title_spacing = title_spacing
        self.scale = scale or Scale()
        self.clipping = clipping or Clipping()
        self.colormap = colormap
        self.colorbar_label = colorbar_label
        self.show_colorbar = show_colorbar
        self.mask_name = mask_name
        self.drawable_patches = drawable_patches or []

        if self.colormap is not None and self.colormap not in self.get_available_colormaps():
            raise ValueError('The given colormap ({}) is not supported.'.format(self.colormap))

    @classmethod
    def _get_attribute_conversions(cls):
        return {'title': StringConversion(),
                'scale': Scale.get_conversion_info(),
                'clipping': Clipping.get_conversion_info(),
                'colormap': StringConversion(),
                'colorbar_label': StringConversion(),
                'title_spacing': FloatConversion(),
                'mask_name': StringConversion(),
                'show_colorbar': BooleanConversion(),
                'drawable_patches': ConvertListElements(DrawablePatches.get_conversion_info())}

    @classmethod
    def get_available_colormaps(cls):
        return get_available_colormaps()

    @classmethod
    def from_yaml(cls, text):
        return cls.get_conversion_info().from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, config_dict):
        return cls.get_conversion_info().from_dict(config_dict)

    def to_dict(self):
        return self.get_conversion_info().to_dict(self)

    def to_yaml(self):
        return yaml.safe_dump(self.get_conversion_info().to_dict(self))

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


class Zoom(SimpleConvertibleConfig):

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
    def _get_attribute_conversions(cls):
        point_converter = Point.get_conversion_info()
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

            new_p0 = Point(np.min([dimensions[0] - p0.y, dimensions[0] - p1.y]), np.min([p0.x, p1.x]))
            new_p1 = Point(np.max([dimensions[0] - p0.y, dimensions[0] - p1.y]), np.max([p0.x, p1.x]))

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


class Point(SimpleConvertibleConfig):

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


class DrawablePatches(SimpleConvertibleConfig):

    def __init__(self, patch_type, patch_args=None, patch_kwargs=None):
        """Container and constructor  for simple drawable patches.

        This class constructs an object in ``matplotlib.patches`` of the given (patch) type with as arguments
        the given args and kwargs.

        Examples:
            DrawablePatches('Rectangle', [xy, width, height], {angle=0.0, ...})

        Args:
            patch_type (str): the type of patch we want to generate. Should be one of the classes in matplotlib.patches.
            patch_args (list_: passed to the constructor of the patch
            patch_kwargs (dict): passed to the constructor of the patch
        """
        self.patch_type = patch_type
        self.patch_args = patch_args or []
        self.patch_kwargs = patch_kwargs or {}

        self._construct_patch()

    @classmethod
    def _get_attribute_conversions(cls):
        return {'patch_type': StringConversion(allow_null=False),
                'patch_args': SimpleListConversion(),
                'patch_kwargs': SimpleDictConversion()}

    def get_patch(self):
        """Get the actual Patch that matplotlib can draw on the images

        Returns:
            patch: the patch to draw on the images
        """
        return self._construct_patch()

    def _construct_patch(self):
        class_type = self._patch_type_to_class(self.patch_type)
        if class_type is None:
            raise ValueError('The given patch type ({}) could not be found, '
                             'please use one of matplotlib.patches.'.format(self.patch_type))
        return class_type(*self.patch_args, **self.patch_kwargs)

    @staticmethod
    def _patch_type_to_class(patch_type):
        import matplotlib.patches
        classes = inspect.getmembers(matplotlib.patches, inspect.isclass)
        for name, class_type in classes:
            if name == patch_type:
                return class_type
        return None


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

    def __init__(self, family='Arial', size=14):
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


class DataInfo(object):
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

    def get_directories(self):
        """Get a list of directories that host all the files in this data info.

        Returns:
            list of str: the list of directories for every file that has an associated directory.
                This list is typically ordered to the number of maps in each directory.
        """
        raise NotImplementedError()

    def get_file_name(self, map_name):
        """Get the file name of the given map

        Returns:
            None if the map could not be found on dir, else a string with the file path.
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
            tuple of Point: two point designating first the upper left corner and second the lower right corner of the
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
    def from_dir(cls, directory):
        if directory is None:
            return cls({}, None)

        if isinstance(directory, (list, tuple)):
            directory = directory[0]

        maps_paths = {}

        for path, map_name, extension in yield_nifti_info(directory):
            if map_name in maps_paths:
                map_name += extension
            maps_paths.update({map_name: path})

        return cls({k: SingleMapInfo(k, load_nifti(v), v) for k, v in maps_paths.items()})

    def get_updated(self, maps):
        """Get a new simple data info object that includes the given new maps.

        In the case of double naming, the old maps are overwritten.

        Args:
            maps (dict): the dictionary with the maps to view, these maps can either be arrays with values,
                nibabel proxy images or SingleMapInfo objects.

        Returns:
            SimpleDataInfo: a new updated data info object
        """
        new_maps = copy(self._input_maps)
        new_maps.update(maps)
        return SimpleDataInfo(new_maps)

    def get_map_names(self):
        return self._input_maps.keys()

    def get_map_data(self, map_name):
        return self.get_single_map_info(map_name).data

    def get_single_map_info(self, map_name):
        value = self._input_maps[map_name]
        if not isinstance(value, SingleMapInfo):
            return SingleMapInfo(map_name, value)
        return value

    def get_directories(self):
        directories = {}

        for map_name in self.get_map_names():
            file_path = self.get_single_map_info(map_name).file_path
            if file_path:
                if os.path.dirname(file_path) in directories:
                    directories.update({os.path.dirname(file_path): directories[os.path.dirname(file_path)]+1})
                else:
                    directories.update({os.path.dirname(file_path): 0})

        return [el[0] for el in sorted(directories.items(), key=lambda x: x[1], reverse=True)]

    def get_file_name(self, map_name):
        return self.get_single_map_info(map_name).file_path

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

        return Point(p0x, p0y), Point(p1x, p1y)


class SingleMapInfo(object):

    def __init__(self, map_name, data, file_path=None):
        """Holds information about a single map.

        Args:
            map_name (str): the name of the map
            data (ndarray or :class:`nibabel.nifti1.Nifti1Image`): the value of the map or the proxy to it
            file_path (str): optionally, the file path with the location of this map
        """
        self._map_name = map_name
        self._data = data
        self._shape = self._data.shape
        self._file_path = file_path

    @property
    def map_name(self):
        return self._map_name

    @property
    def shape(self):
        return self._shape

    @property
    def data(self):
        if isinstance(self._data, nibabel.nifti1.Nifti1Image):
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

        Args:
            mask (ndarray): the mask, we only include elements for which the mask > 0

        Returns:
            tuple: (min, max) the minimum and maximum values
        """
        if mask is not None:
            roi = mdt.create_roi(self.data, mask)
            return roi.min(), roi.max()
        return self.data.min(), self.data.max()

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
