import copy
from mdt.visualization.dict_conversion import ConvertDictElements
from mdt.visualization.maps.base import SingleMapConfig, MapPlotConfig, Zoom, Point


class PlottingFrame(object):

    def __init__(self, controller, plotting_info_viewer=None):
        super(PlottingFrame, self).__init__()
        self._controller = controller
        self._plotting_info_viewer = plotting_info_viewer or NoOptPlottingFrameInfoViewer()

    def set_auto_rendering(self, auto_render):
        """Set if this plotting frame should auto render itself on every configuration update, or not.

        Args:
            auto_render (boolean): if True the plotting frame should auto render, if False it should only
                render on manual updates.
        """

    def redraw(self):
        """Tell the plotting frame to do a redraw."""

    def export_image(self, filename, width, height, dpi=100):
        """Export the current view as an image.

        Args:
            filename (str): where to write the file
            width (int): the width in pixels
            height (int): the height in pixels
            dpi (int): the dpi of the result
        """


class PlottingFrameInfoViewer(object):

    def __init__(self):
        """Implementations of this class can be given to a PlottingFrame to update viewing information.

        As an interface is bridges the gap between the rest of the GUI and the PlottingFrame and
        can encapsulate highlighting interesting aspects of one of the plots.
        """

    def set_voxel_info(self, onscreen_coords, data_index, value):
        """Highlight a single voxel.

        Args:
            onscreen_coords (tuple of x,y): the coordinates of the voxel onscreen
            data_index (tuple of x,y,z,v): the 4d coordinates of the corresponding voxel in the data
            value (float): the value of the object in the 4d coordinates.
        """

    def clear_voxel_info(self):
        """Tell the info viewer that we are no longer looking at a specific voxel."""


class NoOptPlottingFrameInfoViewer(PlottingFrameInfoViewer):

    def set_voxel_info(self, onscreen_coords, data_index, value):
        super(NoOptPlottingFrameInfoViewer, self).set_voxel_info(onscreen_coords, data_index, value)
        print(onscreen_coords, data_index, value)

    def clear_voxel_info(self):
        super(NoOptPlottingFrameInfoViewer, self).clear_voxel_info()
        print('clear')


def cast_value(value, desired_type, alt_value):
    """Cast the given value to the desired type, on failure returns the alternative value.

    Args:
        value (object): the value to cast to the given type
        desired_type (:class:`type`): the type to cast to
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
        if data_info.maps:
            self._validate_maps_to_show(data_info)
            self._validate_dimension(data_info)

            for key in self.__dict__:
                if hasattr(self, '_validate_' + key):
                    getattr(self, '_validate_' + key)(data_info)
        return self

    def _validate_maps_to_show(self, data_info):
        if any(map(lambda k: k not in data_info.maps, self.maps_to_show)):
            raise ValueError('One of the given maps to show is not in the data.')

    def _validate_dimension(self, data_info):
        max_dim = data_info.get_max_dimension(map_names=self.maps_to_show)
        if self.dimension is None or self.dimension > max_dim:
            raise ValueError('The dimension ({}) can not be higher than {}.'.format(self.dimension, max_dim))

    def _validate_slice_index(self, data_info):
        max_slice_index = data_info.get_max_slice_index(self.dimension, map_names=self.maps_to_show)
        if self.slice_index is None or self.slice_index > max_slice_index or self.slice_index < 0:
            raise ValueError('The slice index ({}) can not be higher than '
                             '{} or lower than 0.'.format(self.slice_index, max_slice_index))

    def _validate_volume_index(self, data_info):
        max_volume_index = data_info.get_max_volume_index(map_names=self.maps_to_show)
        if self.volume_index > max_volume_index or self.volume_index < 0:
            raise ValueError('The volume index ({}) can not be higher than '
                             '{} or lower than 0.'.format(self.volume_index, max_volume_index))

    def _validate_zoom(self, data_info):
        max_x = data_info.get_max_x(self.dimension, self.rotate)
        max_y = data_info.get_max_y(self.dimension, self.rotate)

        if self.zoom.p1.x > max_x:
            raise ValueError('The zoom maximum x ({}) can not be larger than {}'.format(self.zoom.p1.x, max_x))

        if self.zoom.p1.y > max_y:
            raise ValueError('The zoom maximum y ({}) can not be larger than {}'.format(self.zoom.p1.y, max_y))

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


class ConfigAction(object):

    def __init__(self):
        """Allows apply and unapply of configuration changes."""
        self._previous_config = None

    def apply(self, data_info, configuration):
        """Apply the changes to the given configuration and return a new one.

        This should return a new configuration with the applied changes and should not update the given configuration.

        By default this method calls _apply(configuration) to facilitate quick implementation.

        Args:
            data_info (DataInfo): the current data information
            configuration (DisplayConfiguration): the configuration object

        Returns:
            ValidatedMapPlotConfig: the updated configuration
        """
        self._previous_config = configuration
        new_config = copy.deepcopy(configuration)
        updated_new_config = self._apply(data_info, new_config)
        if updated_new_config:
            return updated_new_config
        return new_config

    def unapply(self):
        """Return the configuration as it was before the application of this function.

        Returns:
            ValidatedMapPlotConfig: the previous configuration
        """
        return self._previous_config

    def _apply(self, data_info, configuration):
        """Facilitates quick implementation, called by apply()

        One can set configuration changes immediately to the given configuration. If nothing is returned we
        will use the given configuration as the new configuration.

        Args:
            data_info (DataInfo): the current data information
            configuration (ValidatedMapPlotConfig): the configuration object

        Returns:
            GeneralConfiguration or None: the updated configuration. If nothing is returned we use the one given as
                argument.
        """


class SimpleConfigAction(ConfigAction):

    config_attribute = None

    def __init__(self, new_value):
        """A simple configuration action this sets the given value to the config attribute of the configuration."""
        super(SimpleConfigAction, self).__init__()
        self.new_value = new_value

    def _apply(self, data_info, configuration):
        setattr(configuration, self.config_attribute, self.new_value)
        return self._extra_actions(data_info, configuration)

    def _extra_actions(self, data_info, configuration):
        """Called by the default configuration action to apply additional changes"""
        return configuration


class SimpleMapSpecificConfigAction(SimpleConfigAction):

    config_attribute = None

    def __init__(self, map_name, new_value):
        super(SimpleMapSpecificConfigAction, self).__init__(new_value)
        self.map_name = map_name

    def _apply(self, data_info, configuration):
        if self.map_name not in configuration.map_plot_options:
            configuration.map_plot_options[self.map_name] = ValidatedSingleMapConfig()
        single_map_config = super(SimpleMapSpecificConfigAction, self)._apply(
            data_info,
            configuration.map_plot_options[self.map_name])

        if single_map_config is None:
            del configuration.map_plot_options[self.map_name]

        return configuration

    def _extra_actions(self, data_info, configuration):
        single_map_config = super(SimpleMapSpecificConfigAction, self)._extra_actions(data_info, configuration)
        if single_map_config == ValidatedSingleMapConfig():
            return None
        return single_map_config


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
            action (mdt.gui.maps_visualizer.base.ConfigAction): the configuration action to add and apply
        """

    def undo(self):
        """Undo a previous configuration action"""

    def redo(self):
        """Reapply a previously undone configuration action"""

    def has_undo(self):
        """Check if this controller has an undo action available.

        Returns:
            boolean: True if an undo action is available.
        """

    def has_redo(self):
        """Check if this controller has an redo action available.

        Returns:
            boolean: True if an redo action is available.
        """
