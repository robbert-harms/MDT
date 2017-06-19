import copy
from mdt.gui.maps_visualizer.base import SimpleDataConfigModel
from mdt.visualization.maps.base import Zoom, Point, SingleMapConfig


class ModelUpdateAction(object):

    def apply(self, data_config_model):
        """Apply the changes to the given model and return a new model.

        Args:
            data_config_model (mdt.gui.maps_visualizer.base.DataConfigModel): the current data and configuration

        Returns:
            mdt.gui.maps_visualizer.base.DataConfigModel: the updated/new model
        """
        raise NotImplementedError()

    def unapply(self):
        """Return the configuration as it was before the application of this function.

        Returns:
            mdt.gui.maps_visualizer.base.DataConfigModel: the previous model
        """
        raise NotImplementedError()


class NewDataAction(ModelUpdateAction):

    def __init__(self, data, config=None):
        """Sets the new data and (optional) configuration when applied.

        This class will change some parts of the configuration to make sure that removing or adding maps does not
        result in an improper configuration.
        """
        self._previous_model = None
        self._new_config = config
        self._new_data = data

    def apply(self, data_config_model):
        """Apply the changes to the given model and return a new model.

        Args:
            data_config_model (mdt.gui.maps_visualizer.base.DataConfigModel): the current data and configuration

        Returns:
            mdt.gui.maps_visualizer.base.DataConfigModel: the updated/new model
        """
        self._previous_model = data_config_model
        config = self._new_config or self._previous_model.get_config()
        config = copy.deepcopy(config)

        if self._new_data.get_map_names():
            max_dim = self._new_data.get_max_dimension()
            if max_dim < config.dimension:
                config.dimension = max_dim

        config.maps_to_show = list(filter(lambda k: k in self._new_data.get_map_names(), config.maps_to_show))
        if config.mask_name not in self._new_data.get_map_names():
            config.mask_name = None

        return SimpleDataConfigModel(self._new_data, config)

    def unapply(self):
        """Return the configuration as it was before the application of this function.

        Returns:
            mdt.gui.maps_visualizer.base.DataConfigModel: the previous model
        """
        return self._previous_model


class ConfigAction(ModelUpdateAction):

    def __init__(self):
        """An abstract implementation of a model update action meant for updating only the configuration."""
        self._previous_model = None
        self._previous_config = None

    def apply(self, data_config_model):
        """Apply the changes to the configuration and keeps the data intact.

        By default this method calls _apply(data_config_model) to facilitate quick implementation.
        """
        self._previous_model = data_config_model
        self._previous_config = data_config_model.get_config()

        new_config = copy.deepcopy(self._previous_config)
        updated_new_config = self._apply(new_config)

        if updated_new_config:
            return SimpleDataConfigModel(data_config_model.get_data(), updated_new_config)
        return SimpleDataConfigModel(data_config_model.get_data(), new_config)

    def unapply(self):
        """Return the configuration as it was before the application of this function.

        Returns:
            mdt.gui.maps_visualizer.base.DataConfigModel: the previous model
        """
        return self._previous_model

    def _apply(self, config):
        """Facilitates quick implementation, called by apply()

        One can set configuration changes immediately to the configuration in the given model. If None is returned we
        do nothing.

        Args:
            config (mdt.visualization.maps.base.MapPlotConfig): the current data and configuration

        Returns:
            mdt.visualization.maps.base.MapPlotConfig: the updated/new configuration
        """
        raise NotImplementedError()


class SimpleConfigAction(ConfigAction):

    config_attribute = None

    def __init__(self, new_value):
        """A simple configuration action this sets the given value to the config attribute of the configuration."""
        super(SimpleConfigAction, self).__init__()
        self.new_value = new_value

    def _apply(self, configuration):
        setattr(configuration, self.config_attribute, self.new_value)
        return self._extra_actions(configuration)

    def _extra_actions(self, configuration):
        """Called by the default configuration action to apply additional changes"""
        return configuration


class SimpleMapSpecificConfigAction(SimpleConfigAction):

    config_attribute = None

    def __init__(self, map_name, new_value):
        super(SimpleMapSpecificConfigAction, self).__init__(new_value)
        self.map_name = map_name

    def _apply(self, configuration):
        if self.map_name not in configuration.map_plot_options:
            configuration.map_plot_options[self.map_name] = SingleMapConfig()

        single_map_config = super(SimpleMapSpecificConfigAction, self)._apply(
            configuration.map_plot_options[self.map_name])

        if single_map_config is None:
            del configuration.map_plot_options[self.map_name]

        return configuration

    def _extra_actions(self, configuration):
        single_map_config = super(SimpleMapSpecificConfigAction, self)._extra_actions(configuration)
        if single_map_config == SingleMapConfig():
            return None
        return single_map_config


class NewConfigAction(ConfigAction):

    def __init__(self, new_config):
        super(NewConfigAction, self).__init__()
        self.new_config = new_config

    def apply(self, data_config_model):
        self._previous_model = data_config_model
        return SimpleDataConfigModel(data_config_model.get_data(), self.new_config)

    def unapply(self):
        return self._previous_model


class SetDimension(SimpleConfigAction):

    config_attribute = 'dimension'

    def _extra_actions(self, configuration):
        if self.new_value != self._previous_config.dimension:
            data = self._previous_model.get_data()

            max_slice = data.get_max_slice_index(self.new_value, configuration.maps_to_show)
            if configuration.slice_index > max_slice:
                configuration = SetSliceIndex(max_slice // 2).apply(
                    SimpleDataConfigModel(data, configuration)).get_config()

            return SetZoom(Zoom.no_zoom()).apply(SimpleDataConfigModel(data, configuration)).get_config()


class SetSliceIndex(SimpleConfigAction):

    config_attribute = 'slice_index'


class SetVolumeIndex(SimpleConfigAction):

    config_attribute = 'volume_index'


class SetMapsToShow(SimpleConfigAction):

    config_attribute = 'maps_to_show'


class SetColormap(SimpleConfigAction):

    config_attribute = 'colormap'


class SetRotate(SimpleConfigAction):

    config_attribute = 'rotate'

    def _extra_actions(self, configuration):
        if self.new_value != self._previous_config.rotate:

            data = self._previous_model.get_data()

            new_rotation = self.new_value - self._previous_config.rotate
            if self._previous_config.flipud:
                new_rotation *= -1

            new_zoom = self._previous_config.zoom.get_rotated(
                new_rotation,
                data.get_max_x_index(configuration.dimension, self._previous_config.rotate,
                                     configuration.maps_to_show) + 1,
                data.get_max_y_index(configuration.dimension, self._previous_config.rotate,
                                     configuration.maps_to_show) + 1)

            return SetZoom(new_zoom).apply(SimpleDataConfigModel(data, configuration)).get_config()


class SetZoom(SimpleConfigAction):

    config_attribute = 'zoom'


class SetGeneralMask(SimpleConfigAction):

    config_attribute = 'mask_name'


class SetPlotTitle(SimpleConfigAction):

    config_attribute = 'title'


class SetMapTitle(SimpleMapSpecificConfigAction):

    config_attribute = 'title'


class SetMapColorbarLabel(SimpleMapSpecificConfigAction):

    config_attribute = 'colorbar_label'


class SetMapScale(SimpleMapSpecificConfigAction):

    config_attribute = 'scale'


class SetMapClipping(SimpleMapSpecificConfigAction):

    config_attribute = 'clipping'


class SetMapColormap(SimpleMapSpecificConfigAction):

    config_attribute = 'colormap'


class SetFont(SimpleConfigAction):

    config_attribute = 'font'


class SetShowAxis(SimpleConfigAction):

    config_attribute = 'show_axis'


class SetColorBarNmrTicks(SimpleConfigAction):

    config_attribute = 'colorbar_nmr_ticks'


class SetInterpolation(SimpleConfigAction):

    config_attribute = 'interpolation'


class SetFlipud(SimpleConfigAction):

    config_attribute = 'flipud'

    def _extra_actions(self, configuration):
        if self.new_value != self._previous_config.flipud:
            data = self._previous_model.get_data()

            max_y = data.get_max_y_index(configuration.dimension, configuration.rotate,
                                         configuration.maps_to_show) + 1

            new_p0_y = max_y - configuration.zoom.p1.y
            if new_p0_y >= max_y - 1:
                new_p0_y = 0

            new_p1_y = max_y - configuration.zoom.p0.y
            if new_p1_y >= max_y - 1:
                new_p1_y = max_y - 1

            new_zoom = Zoom(Point(configuration.zoom.p0.x, new_p0_y),
                            Point(configuration.zoom.p1.x, new_p1_y))

            return SetZoom(new_zoom).apply(SimpleDataConfigModel(data, configuration)).get_config()
