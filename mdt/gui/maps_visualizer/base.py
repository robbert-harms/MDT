import copy
import matplotlib
import numpy as np

import mdt


class PlottingFrame(object):

    def __init__(self):
        super(PlottingFrame, self).__init__()


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
        for map_name in map_names:
            index = self._map_info[map_name].get_index_first_non_zero_slice(dimension)
            if index is not None:
                return index
        return 0


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


class Diffable(object):

    def __init__(self):
        """An interface for diffable objects. """

    def get_difference(self, other):
        """Get a difference object representing the difference between self and the other diffable.

        This difference object should hold the values of the given other diffable in the case of found differences.

        Args:
            other (Diffable): a diffable compatible with self

        Returns:
            Difference: a difference object representing the differences
        """


class Difference(object):

    def __init__(self):
        """Storage container for the differences between two diffables."""


class GeneralConfigurationDifference(object):

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def __str__(self):
        return str(self.__dict__)


class MapSpecificConfigurationDifference(object):

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def __str__(self):
        return str(self.__dict__)


class GeneralConfiguration(Diffable):

    def __init__(self):
        super(GeneralConfiguration, self).__init__()
        self.dimension = 2
        self.slice_index = 0
        self.volume_index = 0

        # todo implement: zoom, maps_to_show (ordering), map_plot_options

        self.zoom = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
        self.maps_to_show = None
        self.colormap = 'hot'
        self.rotate = 0
        self.map_plot_options = {}

    @classmethod
    def from_dict(cls, config_dict):
        """Create and return a new instance using the given configuration dict.

        The layout and items of the config dict should match those from the function 'to_dict'

        Args:
            config_dict (dict): the new configuration dictionary
        """
        config = GeneralConfiguration()
        config.__dict__.update(config_dict)
        config.map_plot_options = {key: MapSpecificConfiguration.from_dict(value)
                                   for key, value in config_dict['map_plot_options'].items()}
        return config

    def validate(self, data_info):
        """Validate this config using the given data information.

        This will always return a copy of this configuration with validated settings.

        Args:
            data_info (DataInfo): the data information used to create a valid copy of this configuration.

        Returns:
            GeneralConfiguration: new configuration with validated settings.
        """
        validated = copy.deepcopy(self)
        validated.maps_to_show = [key for key in validated.maps_to_show if key in data_info.maps]

        if validated.rotate not in [0, 90, 180, 270]:
            validated.rotate = 0

        try:
            matplotlib.cm.get_cmap(validated.colormap)
        except ValueError:
            validated.colormap = 'hot'

        if validated.dimension is None:
            validated.dimension = 2
        else:
            validated.dimension = min(validated.dimension, data_info.get_max_dimension(validated.maps_to_show))

        if validated.slice_index is None:
            validated.slice_index = data_info.get_index_first_non_zero_slice(validated.dimension,
                                                                             validated.maps_to_show)
        else:
            max_slice_index = data_info.get_max_slice_index(validated.dimension, validated.maps_to_show)
            if validated.slice_index > max_slice_index:
                validated.slice_index = data_info.get_index_first_non_zero_slice(validated.dimension,
                                                                                 validated.maps_to_show)

        if validated.volume_index is None:
            validated.volume_index = 0
        else:
            validated.volume_index = min(validated.volume_index, data_info.get_max_volume_index(validated.maps_to_show))

        return validated

    def get_difference(self, other):
        differences = {}

        for key, value in self.__dict__.items():
            if key not in ['map_plot_options']:
                if value != getattr(other, key):
                    differences.update({key: getattr(other, key)})

        if self.map_plot_options != other.map_plot_options:
            map_diffs = {}

            for key, value in other.map_plot_options.items():
                if key not in self.map_plot_options:
                    map_diffs.update(key=MapSpecificConfiguration().get_difference(value))
                else:
                    diff = self.map_plot_options[key].get_difference(value)
                    if diff:
                        map_diffs.update(key=diff)

            if map_diffs:
                differences['map_plot_options'] = map_diffs

        if differences:
            return GeneralConfigurationDifference(**differences)

    def to_dict(self):
        """Get the whole configuration as a multi-level dictionary.

        This can be useful for converting the configuration to a string.
        """
        result = copy.copy(self.__dict__)
        result['map_plot_options'] = {key: value.to_dict() for key, value in self.map_plot_options.items()}
        return result


class MapSpecificConfiguration(Diffable):

    def __init__(self, title=None, scale=None, clipping=None, colormap=None):
        super(MapSpecificConfiguration, self).__init__()
        self.title = title
        self.scale = scale or {'min': None, 'max': None}
        self.clipping = clipping or {'min': None, 'max': None}
        self.colormap = colormap

    @classmethod
    def from_dict(cls, config_dict):
        """Create and return a new instance using the given configuration dict.

        The layout and items of the config dict should match those from the function 'to_dict'

        Args:
            config_dict (dict): the new configuration dictionary
        """
        config = MapSpecificConfiguration()
        config.__dict__.update(config_dict)
        return config

    def get_difference(self, other):
        differences = {}
        for key, value in self.__dict__.items():
            if value != getattr(other, key):
                differences.update({key: getattr(other, key)})

        if differences:
            return MapSpecificConfigurationDifference(**differences)

    def to_dict(self):
        return copy.copy(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


class ConfigAction(object):

    def __init__(self):
        """Allows apply and unapply of configuration changes."""
        self._previous_config = None

    def apply(self, configuration):
        """Apply the changes to the given configuration and return a new one.

        This should return a new configuration with the applied changes and should not update the given configuration.

        By default this method calls _apply(configuration) to facilitate quick implementation.

        Args:
            configuration (GeneralConfiguration): the configuration object

        Returns:
            GeneralConfiguration: the updated configuration
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
            GeneralConfiguration: the previous configuration
        """
        return self._previous_config

    def _apply(self, configuration):
        """Facilitates quick implementation, called by apply()

        One can set configuration changes immediately to the given configuration. If nothing is returned we
        will use the given configuration as the new configuration.

        Args:
            configuration (GeneralConfiguration): the configuration object

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
            item.update(self.new_value)
        else:
            setattr(configuration, self.config_attribute, self.new_value)


class SimpleMapSpecificConfigAction(ConfigAction):

    config_attribute = None
    use_update = False

    def __init__(self, map_name, new_value):
        super(SimpleMapSpecificConfigAction, self).__init__()
        self.map_name = map_name
        self.new_value = new_value

    def _apply(self, configuration):
        if self.map_name not in configuration.map_plot_options:
            configuration.map_plot_options[self.map_name] = MapSpecificConfiguration()

        if self.use_update:
            item = getattr(configuration.map_plot_options[self.map_name], self.config_attribute)
            item.update(self.new_value)
        else:
            setattr(configuration.map_plot_options[self.map_name], self.config_attribute, self.new_value)


class Controller(object):

    def __init__(self):
        """Controller interface"""
        super(Controller, self).__init__()

    def set_data(self, data_info, config=None):
        """Set new data to visualize.

        Args:
            data_info (DataInfo): the new data to visualize
            config (GeneralConfiguration): the new configuration for the data
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
            general_config (GeneralConfiguration): the general configuration
        """

    def get_config(self):
        """Get the current configuration.

        Returns:
            GeneralConfiguration: the current general configuration.
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
