from mdt.visualization.maps.base import Zoom, Point
from .base import ConfigAction, SimpleConfigAction, SimpleMapSpecificConfigAction


class SetDimension(SimpleConfigAction):

    config_attribute = 'dimension'

    def _extra_actions(self, data_info, configuration):
        if self.new_value != self._previous_config.dimension:
            max_slice = data_info.get_max_slice_index(self.new_value, configuration.maps_to_show)
            if configuration.slice_index > max_slice:
                configuration = SetSliceIndex(max_slice // 2).apply(data_info, configuration)
            return SetZoom(Zoom.no_zoom()).apply(data_info, configuration)


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

    def _extra_actions(self, data_info, configuration):
        if self.new_value != self._previous_config.rotate:

            new_rotation = self.new_value - self._previous_config.rotate
            if self._previous_config.flipud:
                new_rotation *= -1

            new_zoom = self._previous_config.zoom.get_rotated(
                new_rotation,
                data_info.get_max_x_index(configuration.dimension, self._previous_config.rotate,
                                          configuration.maps_to_show) + 1,
                data_info.get_max_y_index(configuration.dimension, self._previous_config.rotate,
                                          configuration.maps_to_show) + 1)

            return SetZoom(new_zoom).apply(data_info, configuration)


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


class NewConfigAction(ConfigAction):

    def __init__(self, new_config):
        super(NewConfigAction, self).__init__()
        self.new_config = new_config

    def apply(self, data_info, configuration):
        self._previous_config = configuration
        return self.new_config

    def unapply(self):
        return self._previous_config


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

    def _extra_actions(self, data_info, configuration):
        if self.new_value != self._previous_config.flipud:
            max_y = data_info.get_max_y_index(configuration.dimension, configuration.rotate,
                                              configuration.maps_to_show) + 1

            new_p0_y = max_y - configuration.zoom.p1.y
            if new_p0_y >= max_y - 1:
                new_p0_y = 0

            new_p1_y = max_y - configuration.zoom.p0.y
            if new_p1_y >= max_y - 1:
                new_p1_y = max_y - 1

            new_zoom = Zoom(Point(configuration.zoom.p0.x, new_p0_y),
                            Point(configuration.zoom.p1.x, new_p1_y))

            return SetZoom(new_zoom).apply(data_info, configuration)
