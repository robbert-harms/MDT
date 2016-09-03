import copy
import matplotlib

from mdt.visualization.dict_conversion import ConvertDictElements
from mdt.visualization.maps.base import SingleMapConfig, MapPlotConfig, Zoom, Point


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


class ValidatedMapPlotConfig(MapPlotConfig):

    @classmethod
    def _get_attribute_conversions(cls):
        conversions = super(ValidatedMapPlotConfig, cls)._get_attribute_conversions()
        conversions['map_plot_options'] = ConvertDictElements(ValidatedSingleMapConfig.get_conversion_info())
        return conversions

    def validate(self, data_info):
        self._validate_maps_to_show(data_info)
        for key in self.__dict__:
            if hasattr(self, '_validate_' + key):
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
        p1x_val = self.zoom.p1.x
        if self.zoom.p1.x == 0:
            try:
                p1x_val = data_info.get_max_x(self.dimension, self.rotate, self.maps_to_show)
            except ValueError:
                pass

        p1y_val = self.zoom.p1.y
        if self.zoom.p1.y == 0:
            try:
                p1y_val = data_info.get_max_y(self.dimension, self.rotate, self.maps_to_show)
            except ValueError:
                pass

        self.zoom = Zoom(self.zoom.p0, Point(p1x_val, p1y_val))

    def _validate_map_plot_options(self, data_info):
        for key in self.map_plot_options:
            if key not in data_info.maps:
                del self.map_plot_options[key]

        for key, value in self.map_plot_options.items():
            if value is not None:
                self.map_plot_options[key] = value.validate(data_info)


class ValidatedSingleMapConfig(SingleMapConfig):

    def validate(self, data_info):
        for key in self.__dict__:
            if hasattr(self, '_validate_' + key):
                getattr(self, '_validate_' + key)(data_info)
        return self

    def _validate_colormap(self, data_info):
        if self.colormap:
            try:
                matplotlib.cm.get_cmap(self.colormap)
            except ValueError:
                self.colormap = None


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
            ValidatedMapPlotConfig: the updated configuration
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
            ValidatedMapPlotConfig: the previous configuration
        """
        return self._previous_config

    def _apply(self, configuration):
        """Facilitates quick implementation, called by apply()

        One can set configuration changes immediately to the given configuration. If nothing is returned we
        will use the given configuration as the new configuration.

        Args:
            configuration (ValidatedMapPlotConfig): the configuration object

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
            configuration.map_plot_options[self.map_name] = ValidatedSingleMapConfig()
        super(SimpleMapSpecificConfigAction, self)._apply(configuration.map_plot_options[self.map_name])
        return configuration


class Controller(object):

    def __init__(self):
        """Controller interface"""
        super(Controller, self).__init__()

    def set_data(self, data_info, config=None):
        """Set new data to visualize.

        Args:
            data_info (mdt.visualization.maps.base.DataInfo): the new data to visualize
            config (ValidatedMapPlotConfig): the new configuration for the data
                If given, we will display the new data immediately with the given config
        """

    def get_data(self):
        """Get the current data.

        Returns:
            mdt.visualization.maps.base.DataInfo: the current data information
        """

    def set_config(self, general_config):
        """Set the general configuration to the given config.

        Setting this should automatically update all the listeners.

        Args:
            general_config (ValidatedMapPlotConfig): the general configuration
        """

    def get_config(self):
        """Get the current configuration.

        Returns:
            ValidatedMapPlotConfig: the current general configuration.
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
