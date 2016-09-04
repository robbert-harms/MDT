from mdt.visualization.maps.base import Zoom, Point
from .base import ConfigAction, SimpleConfigAction, SimpleMapSpecificConfigAction


class SetDimension(SimpleConfigAction):

    config_attribute = 'dimension'

    def _extra_actions(self, configuration):
        if self.new_value != self._previous_config.dimension:
            return SetZoom(Zoom(Point(0, 0), Point(0, 0))).apply(configuration)


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
        if self.new_value != self._previous_config.dimension:
            return SetZoom(Zoom(Point(0, 0), Point(0, 0))).apply(configuration)


class SetZoom(SimpleConfigAction):

    config_attribute = 'zoom'


class SetMapTitle(SimpleMapSpecificConfigAction):

    config_attribute = 'title'


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

    def apply(self, configuration):
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
