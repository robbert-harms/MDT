import copy
import matplotlib
import numpy as np
import mdt


class PlottingFrame(object):

    def __init__(self, controller):
        super(PlottingFrame, self).__init__()
        self._controller = controller

    def export_image(self, filename, width, height, dpi=100):
        """Export the current view as an image.

        Args:
            filename (str): where to write the file
            width (int): the width in pixels
            height (int): the height in pixels
            dpi (int): the dpi of the result
        """


class DataInfo(object):

    def __init__(self, maps, directory=None):
        """A container for basic information about the volume maps we are viewing.

        Args:
            maps (dict): the dictionary with the maps to view
            directory (str): the directory from which the maps where loaded
        """
        self.maps = maps
        self.directory = directory
        self._map_info = {key: SingleMapInfo(key, value) for key, value in self.maps.items()}
        self.sorted_keys = list(sorted(maps.keys()))

    @classmethod
    def from_dir(cls, directory):
        return DataInfo(mdt.load_volume_maps(directory), directory)

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
        return max(self._map_info[map_name].max_dimension() for map_name in map_names)

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
        return max(self._map_info[map_name].max_slice_index(dimension) for map_name in map_names)

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
        return max(self._map_info[map_name].max_volume_index() for map_name in map_names)

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
            index = self._map_info[map_name].get_index_first_non_zero_slice(dimension)
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
        return min(self._map_info[map_name].get_max_x(dimension, rotate) for map_name in map_names)

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
        return min(self._map_info[map_name].get_max_y(dimension, rotate) for map_name in map_names)


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


class DisplayConfigurationInterface(object):

    @classmethod
    def from_dict(cls, config_dict):
        """Create and return a new instance using the given configuration dict.

        The layout and items of the config dict should match those from the function 'to_dict'

        Args:
            config_dict (dict): the new configuration dictionary
        """

    def to_dict(self):
        """Get the whole configuration as a multi-level dictionary.

        This can be useful for converting the configuration to a string.

        Returns:
            dict: a 'serialized' version of this class
        """

    def validate(self, data_info):
        """Validate this config using the given data information.

        This will change the values in place. If a copy is needed the calling class must do that.

        Args:
            data_info (DataInfo): the data information used to create a valid copy of this configuration.

        Returns:
            DisplayConfigurationInterface: new configuration with validated settings.
        """


def cast_value(value, desired_type, alt_value):
    """Cast the given value to the desired type, on failure returns the alternative value.

    Args:
        value (object): the value to cast to the given type
        desired_type (type): the type to cast to
        alt_value (object): the alternative value if casting threw exceptions

    Returns:
        the desired casted value or the alternative value if casting failed.
    """
    try:
        return desired_type(value)
    except TypeError:
        return alt_value
    except ValueError:
        return alt_value


class DisplayConfiguration(DisplayConfigurationInterface):

    def __init__(self):
        super(DisplayConfiguration, self).__init__()
        self.dimension = 2
        self.slice_index = 0
        self.volume_index = 0
        self.zoom = {'x_0': 0, 'y_0': 0, 'x_1': 0, 'y_1': 0}
        self.maps_to_show = []
        self.colormap = 'hot'
        self.rotate = 90
        self.font_size = 14
        self.show_axis = True
        self.colorbar_nmr_ticks = 10
        self.map_plot_options = {} # todo implement in GUI: map_plot_options, add option for GridLayout

    @classmethod
    def from_dict(cls, config_dict):
        if config_dict is None:
            config_dict = {}

        config = DisplayConfiguration()
        config.__dict__.update(config_dict)
        config.map_plot_options = {key: MapSpecificConfiguration.from_dict(value)
                                   for key, value in config_dict.get('map_plot_options', {}).items()}
        return config

    def to_dict(self):
        result = copy.copy(self.__dict__)
        result['map_plot_options'] = {key: value.to_dict() for key, value in self.map_plot_options.items()
                                      if value is not None}
        return result

    def validate(self, data_info):
        self._validate_maps_to_show(data_info)
        for key in self.__dict__:
            getattr(self, '_validate_' + key)(data_info)
        return self

    def _validate_maps_to_show(self, data_info):
        if self.maps_to_show:
            self.maps_to_show = [key for key in self.maps_to_show if key in data_info.maps]
        else:
            self.maps_to_show = data_info.sorted_keys

    def _validate_rotate(self, data_info):
        if self.rotate not in [0, 90, 180, 270]:
            self.rotate = 0

    def _validate_colormap(self, data_info):
        try:
            matplotlib.cm.get_cmap(self.colormap)
        except ValueError:
            self.colormap = 'hot'

    def _validate_dimension(self, data_info):
        if self.dimension is None:
            self.dimension = 2
        else:
            self.dimension = cast_value(self.dimension, int, 0)
            try:
                self.dimension = min(self.dimension, data_info.get_max_dimension(self.maps_to_show))
            except ValueError:
                self.dimension = 2

    def _validate_slice_index(self, data_info):
        try:
            if self.slice_index is None:
                self.slice_index = data_info.get_index_first_non_zero_slice(self.dimension, self.maps_to_show)
            else:
                self.slice_index = cast_value(self.slice_index, int, 0)
                max_slice_index = data_info.get_max_slice_index(self.dimension, self.maps_to_show)
                if self.slice_index > max_slice_index:
                    self.slice_index = data_info.get_index_first_non_zero_slice(self.dimension, self.maps_to_show)
        except ValueError:
            self.slice_index = 0

    def _validate_volume_index(self, data_info):
        if self.volume_index is None:
            self.volume_index = 0
        else:
            self.volume_index = cast_value(self.volume_index, int, 0)
            try:
                self.volume_index = min(self.volume_index, data_info.get_max_volume_index(self.maps_to_show))
            except ValueError:
                self.volume_index = 0

    def _validate_zoom(self, data_info):
        if self.zoom is None:
            self.zoom = {'x_0': 0, 'y_0': 0, 'x_1': 0, 'y_1': 0}
        else:
            for item in 'x_0', 'x_1', 'y_0', 'y_1':
                if item not in self.zoom:
                    self.zoom.update({item: 0})
                self.zoom[item] = cast_value(self.zoom[item], int, 0)

        if self.zoom['x_1'] == 0:
            try:
                self.zoom['x_1'] = data_info.get_max_x(self.dimension, self.rotate, self.maps_to_show)
            except ValueError:
                self.zoom['x_1'] = 0

        if self.zoom['y_1'] == 0:
            try:
                self.zoom['y_1'] = data_info.get_max_y(self.dimension, self.rotate, self.maps_to_show)
            except ValueError:
                self.zoom['y_1'] = 0

    def _validate_map_plot_options(self, data_info):
        for key in self.map_plot_options:
            if key not in data_info.maps:
                del self.map_plot_options[key]

        for key, value in self.map_plot_options.items():
            if value is not None:
                self.map_plot_options[key] = value.validate(data_info)

    def _validate_font_size(self, data_info):
        self.font_size = cast_value(self.font_size, int, 14)

    def _validate_show_axis(self, data_info):
        self.show_axis = cast_value(self.show_axis, bool, True)

    def _validate_colorbar_nmr_ticks(self, data_info):
        self.colorbar_nmr_ticks = cast_value(self.colorbar_nmr_ticks, int, None)

    def __eq__(self, other):
        if not isinstance(other, DisplayConfiguration):
            return NotImplemented
        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class MapSpecificConfiguration(DisplayConfigurationInterface):

    def __init__(self, title=None, scale=None, clipping=None, colormap=None):
        super(MapSpecificConfiguration, self).__init__()
        self.title = title
        self.scale = scale or {'min': None, 'max': None}
        self.clipping = clipping or {'min': None, 'max': None}
        self.colormap = colormap

    @classmethod
    def from_dict(cls, config_dict):
        config = MapSpecificConfiguration()
        config.__dict__.update(config_dict)
        return config

    def to_dict(self):
        return copy.copy(self.__dict__)

    def validate(self, data_info):
        for key in self.__dict__:
            getattr(self, '_validate_' + key)(data_info)
        return self

    def _validate_scale(self, data_info):
        if self.scale is None:
            self.scale = {'min': None, 'max': None}
        else:
            for item in 'min', 'max':
                if item not in self.scale:
                    self.scale.update({item: None})
                self.scale[item] = cast_value(self.scale[item], int, None)

    def _validate_clipping(self, data_info):
        if self.clipping is None:
            self.clipping = {'min': None, 'max': None}
        else:
            for item in 'min', 'max':
                if item not in self.clipping:
                    self.clipping.update({item: None})
                self.clipping[item] = cast_value(self.clipping[item], int, None)

    def _validate_title(self, data_info):
        self.title = cast_value(self.title, str, None)

    def _validate_colormap(self, data_info):
        if self.colormap:
            try:
                matplotlib.cm.get_cmap(self.colormap)
            except ValueError:
                self.colormap = None

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, MapSpecificConfiguration):
            return NotImplemented

        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class ConfigAction(object):

    def __init__(self):
        """Allows apply and unapply of configuration changes."""
        self._previous_config = None

    def apply(self, configuration):
        """Apply the changes to the given configuration and return a new one.

        This should return a new configuration with the applied changes and should not update the given configuration.

        By default this method calls _apply(configuration) to facilitate quick implementation.

        Args:
            configuration (DisplayConfiguration): the configuration object

        Returns:
            DisplayConfiguration: the updated configuration
        """
        self._previous_config = configuration
        new_config = copy.deepcopy(configuration)
        updated_new_config = self._apply(new_config)
        if updated_new_config:
            return updated_new_config
        return new_config

    def unapply(self):
        """Return the configuration as it was before the application of this function.

        Returns:
            DisplayConfiguration: the previous configuration
        """
        return self._previous_config

    def _apply(self, configuration):
        """Facilitates quick implementation, called by apply()

        One can set configuration changes immediately to the given configuration. If nothing is returned we
        will use the given configuration as the new configuration.

        Args:
            configuration (DisplayConfiguration): the configuration object

        Returns:
            GeneralConfiguration or None: the updated configuration. If nothing is returned we use the one given as
                argument.
        """


class SimpleConfigAction(ConfigAction):

    config_attribute = None
    use_update = False

    def __init__(self, new_value):
        """A simple configuration action this sets the given value to the config attribute of the configuration."""
        super(SimpleConfigAction, self).__init__()
        self.new_value = new_value

    def _apply(self, configuration):
        if self.use_update:
            item = getattr(configuration, self.config_attribute)
            if self.new_value is None:
                setattr(configuration, self.config_attribute, None)
            else:
                item.update(self.new_value)
        else:
            setattr(configuration, self.config_attribute, self.new_value)

        return self._extra_actions(configuration)

    def _extra_actions(self, configuration):
        """Called by the default configuration action to apply additional changes"""
        return configuration


class SimpleMapSpecificConfigAction(SimpleConfigAction):

    config_attribute = None
    use_update = False

    def __init__(self, map_name, new_value):
        super(SimpleMapSpecificConfigAction, self).__init__(new_value)
        self.map_name = map_name

    def _apply(self, configuration):
        if self.map_name not in configuration.map_plot_options:
            configuration.map_plot_options[self.map_name] = MapSpecificConfiguration()
        return self._apply(configuration.map_plot_options[self.map_name])


class Controller(object):

    def __init__(self):
        """Controller interface"""
        super(Controller, self).__init__()

    def set_data(self, data_info, config=None):
        """Set new data to visualize.

        Args:
            data_info (DataInfo): the new data to visualize
            config (DisplayConfiguration): the new configuration for the data
                If given, we will display the new data immediately with the given config
        """

    def get_data(self):
        """Get the current data.

        Returns:
            DataInfo: the current data information
        """

    def set_config(self, general_config):
        """Set the general configuration to the given config.

        Setting this should automatically update all the listeners.

        Args:
            general_config (DisplayConfiguration): the general configuration
        """

    def get_config(self):
        """Get the current configuration.

        Returns:
            DisplayConfiguration: the current general configuration.
        """

    def apply_action(self, action):
        """Apply a new configuration action.

        If there is no difference between the current config and the one generated by this new action, the
        action will not be stored in history and will not need to be applied.

        Args:
            action (ConfigAction): the configuration action to add and apply
        """

    def undo(self):
        """Undo a previous configuration action"""

    def redo(self):
        """Reapply a previously undone configuration action"""
