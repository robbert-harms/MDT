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
        if self.new_value != self._previous_config.dimension:
            # new_zoom = self._previous_config.zoom.rotate(
            #     self.new_value,
            #     data_info.get_max_x(configuration.dimension, 0, configuration.maps_to_show),
            #     data_info.get_max_y(configuration.dimension, 0, configuration.maps_to_show))

            # return SetZoom(new_zoom).apply(data_info, configuration)
            #todo
            return SetZoom(Zoom.no_zoom()).apply(data_info, configuration)


class SetZoom(SimpleConfigAction):

    config_attribute = 'zoom'


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
            max_y = data_info.get_max_y(configuration.dimension, configuration.rotate,
                                        configuration.maps_to_show)

            new_zoom = Zoom(Point(configuration.zoom.p0.x, max_y - configuration.zoom.p1.y),
                            Point(configuration.zoom.p1.x, max_y - configuration.zoom.p0.y))

            return SetZoom(new_zoom).apply(data_info, configuration)
