from .base import ConfigAction, SimpleConfigAction, SimpleMapSpecificConfigAction, MapSpecificConfiguration


class SetDimension(ConfigAction):

    def __init__(self, dimension):
        super(SetDimension, self).__init__()
        self.dimension = dimension

    def _apply(self, configuration):
        configuration.dimension = self.dimension

        if self.dimension != self._previous_config.dimension:
            return SetZoom({'x': 0, 'y': 0, 'w': 0, 'h': 0}).apply(configuration)


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


class SetMapColormapp(SimpleMapSpecificConfigAction):

    config_attribute = 'colormap'
