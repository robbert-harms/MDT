from .base import ConfigAction, SimpleConfigAction, SimpleMapSpecificConfigAction, \
    DisplayConfiguration


class SetDimension(SimpleConfigAction):

    config_attribute = 'dimension'

    def _extra_actions(self, configuration):
        if self.new_value != self._previous_config.dimension:
            return SetZoom(None).apply(configuration)


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
            return SetZoom(None).apply(configuration)


class SetZoom(SimpleConfigAction):

    config_attribute = 'zoom'
    use_update = True


class SetMapTitle(SimpleMapSpecificConfigAction):

    config_attribute = 'title'


class SetMapScale(SimpleMapSpecificConfigAction):

    config_attribute = 'scale'
    use_update = True


class SetMapClipping(SimpleMapSpecificConfigAction):

    config_attribute = 'clipping'
    use_update = True


class SetMapColormap(SimpleMapSpecificConfigAction):

    config_attribute = 'colormap'


class FromDictAction(ConfigAction):

    def __init__(self, config_dict):
        super(FromDictAction, self).__init__()
        self.config_dict = config_dict

    def apply(self, configuration):
        self._previous_config = configuration
        return DisplayConfiguration.from_dict(self.config_dict)

    def unapply(self):
        return self._previous_config


class SetFontSize(SimpleConfigAction):

    config_attribute = 'font_size'


class SetShowAxis(SimpleConfigAction):

    config_attribute = 'show_axis'


class SetColorBarNmrTicks(SimpleConfigAction):

    config_attribute = 'colorbar_nmr_ticks'
