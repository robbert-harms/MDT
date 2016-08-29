import copy


class ImagesInfo():

    def __init__(self):
        """A container for basic information about the images we are viewing."""

    def get_max_volume(self):
        pass

    def get_max_dimension(self):
        pass

    def get_max_slice_index(self, dimension):
        pass


class PlottingFrame(object):

    def __init__(self):
        super(PlottingFrame, self).__init__()

    def set_dimension(self, dimension):
        """Set the dimension of the plots to the given dimension.

        This only accepts the values 0, 1 and 2 being the three dimensions we visualize. To visualize
        a different volume use the function set_volume_index.

        Args:
            dimension (int): the dimension to set the plots to
        """

    def set_slice_index(self, slice_index):
        """Set the slice index in the current dimension to this value.

        Args:
            slice_index (int): the new slice index
        """


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
        self.current_directory = None
        self.maps = {}
        self.dimension = None
        self.slice_index = None
        self.volume_index = None
        self.zoom = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
        self.maps_to_show = None
        self.colormap = 'hot'
        self.rotate = 0
        self.map_plot_options = {}

    def get_difference(self, other):
        differences = {}
        for item in ['maps', 'current_directory', 'dimension', 'slice_index', 'volume_index', 'colormap',
                     'rotate', 'maps_to_show', 'zoom']:
            if getattr(self, item) != getattr(other, item):
                differences.update({item: getattr(other, item)})

        if self.map_plot_options != other.map_plot_options:
            map_diffs = {}

            for key, value in other.map_plot_options.items():
                if key not in self.map_plot_options:
                    map_diffs.update(key=MapSpecificConfiguration().get_difference(value))
                else:
                    map_diffs.update(key=self.map_plot_options[key].get_difference(value))

            differences['map_plot_options'] = map_diffs

        if differences:
            return GeneralConfigurationDifference(**differences)

    def get_dict(self):
        """Get the whole configuration as a multi-level dictionary.

        This can be useful for converting the configuration to a string.
        """
        #todo

    def __copy__(self):
        config_copy = GeneralConfiguration()
        config_copy.__dict__ = copy.copy(self.__dict__)
        config_copy.zoom = copy.copy(self.zoom)
        config_copy.maps_to_show = copy.copy(self.maps_to_show)
        config_copy.map_plot_options = copy.deepcopy(self.map_plot_options)
        return config_copy


class MapSpecificConfiguration(Diffable):

    def __init__(self):
        super(MapSpecificConfiguration, self).__init__()
        self.title = None
        self.scale = {'min': None, 'max': None}
        self.clipping = {'min': None, 'max': None}
        self.colormap = None

    def get_difference(self, other):
        differences = {}
        for item in self.__dict__.keys():
            if getattr(self, item) != getattr(other, item):
                differences.update({item: getattr(other, item)})

        if differences:
            return MapSpecificConfigurationDifference(**differences)

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
        new_config = copy.copy(configuration)
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

    def add_action(self, action):
        """Add and apply a new configuration action.

        Args:
            action (ConfigAction): the configuration action to add and apply
        """

    def undo(self):
        """Undo a previous configuration action"""

    def redo(self):
        """Reapply a previously undone configuration action"""
