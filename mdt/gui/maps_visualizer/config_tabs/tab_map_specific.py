from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget

from mdt.gui.maps_visualizer.actions import SetMapTitle, SetMapColormap, SetMapScale, SetMapClipping, \
    SetMapColorbarLabel
from mdt.gui.maps_visualizer.base import DataConfigModel
from mdt.gui.maps_visualizer.design.ui_MapSpecificOptions import Ui_MapSpecificOptions
from mdt.gui.maps_visualizer.design.ui_TabMapSpecific import Ui_TabMapSpecific
from mdt.gui.utils import blocked_signals, TimedUpdate, split_long_path_elements
from mdt.visualization.maps.base import DataInfo, SingleMapConfig, MapPlotConfig

__author__ = 'Robbert Harms'
__date__ = "2016-09-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TabMapSpecific(QWidget, Ui_TabMapSpecific):

    def __init__(self, controller, parent=None):
        super(TabMapSpecific, self).__init__(parent)
        self.setupUi(self)

        self.map_specific_tab = MapSpecificOptions(controller, self)
        self.mapSpecificOptionsPosition.addWidget(self.map_specific_tab)

        self._controller = controller
        self._controller.model_updated.connect(self.model_updated)

        self.selectedMap.currentIndexChanged.connect(
            lambda ind: self._update_map_specifics(self.selectedMap.itemData(ind, Qt.UserRole)))

    @pyqtSlot(DataConfigModel)
    def model_updated(self, model):
        config = model.get_config()
        map_names = config.maps_to_show

        with blocked_signals(self.selectedMap):
            current_selected = self.selectedMap.currentData(Qt.UserRole)

            self.selectedMap.clear()
            self.selectedMap.addItems(map_names)

            for index, map_name in enumerate(map_names):
                self.selectedMap.setItemData(index, map_name, Qt.UserRole)

                if map_name in config.map_plot_options and config.map_plot_options[map_name].title:
                    title = config.map_plot_options[map_name].title
                    self.selectedMap.setItemData(index, map_name + ' (' + title + ')', Qt.DisplayRole)

            for ind in range(self.selectedMap.count()):
                if self.selectedMap.itemData(ind, Qt.UserRole) == current_selected:
                    self.selectedMap.setCurrentIndex(ind)
                    break

        if self.selectedMap.count():
            self._update_map_specifics(self.selectedMap.currentData(Qt.UserRole))
        else:
            self._update_map_specifics(None)

    def _update_map_specifics(self, map_name):
        """Set the map specific options to reflect the settings of the given map"""
        if map_name is None:
            self.map_specific_tab.reset()
        else:
            self.map_specific_tab.use(map_name)


class MapSpecificOptions(QWidget, Ui_MapSpecificOptions):

    def __init__(self, controller, parent=None):
        super(MapSpecificOptions, self).__init__(parent)
        self.setupUi(self)
        self._controller = controller

        current_model = self._controller.get_model()

        self._current_map = None
        self.colormap.addItems(['-- Use global --'] + current_model.get_config().get_available_colormaps())
        self.colormap.currentIndexChanged.connect(self._update_colormap)
        self.data_clipping_min.valueChanged.connect(self._update_clipping_min)
        self.data_clipping_max.valueChanged.connect(self._update_clipping_max)
        self.data_scale_min.valueChanged.connect(self._update_scale_min)
        self.data_scale_max.valueChanged.connect(self._update_scale_max)

        self.data_scale_max.set_update_delay(500)
        self.data_scale_min.set_update_delay(500)
        self.data_clipping_max.set_update_delay(500)
        self.data_clipping_min.set_update_delay(500)

        self.data_set_use_scale.stateChanged.connect(self._set_use_scale)
        self.use_data_scale_min.stateChanged.connect(self._set_use_data_scale_min)
        self.use_data_scale_max.stateChanged.connect(self._set_use_data_scale_max)

        self.data_set_use_clipping.stateChanged.connect(self._set_use_clipping)
        self.use_data_clipping_min.stateChanged.connect(self._set_use_data_clipping_min)
        self.use_data_clipping_max.stateChanged.connect(self._set_use_data_clipping_max)

        self._title_timer = TimedUpdate(self._update_map_title)
        self.map_title.textChanged.connect(lambda v: self._title_timer.add_delayed_callback(500, v))

        self._colorbar_label_timer = TimedUpdate(self._update_colorbar_label)
        self.data_colorbar_label.textChanged.connect(lambda v: self._colorbar_label_timer.add_delayed_callback(500, v))

        self.info_Clipping.set_collapse(True)

        self._auto_enable_scale_min = False
        self._auto_enable_scale_max = False
        self._auto_enable_clipping_min = False
        self._auto_enable_clipping_max = False

        self.reset()

    def reset(self):
        """Set all the values to their defaults"""
        self._current_map = None
        self.colormap.setCurrentText('hot')

        with blocked_signals(self.map_title):
            self.map_title.setText('')

        with blocked_signals(self.data_colorbar_label):
            self.data_colorbar_label.setText('')

        with blocked_signals(self.data_clipping_min):
            self.data_clipping_min.setValue(0)

        with blocked_signals(self.data_clipping_max):
            self.data_clipping_max.setValue(0)

        with blocked_signals(self.data_scale_min):
            self.data_scale_min.setValue(0)

        with blocked_signals(self.data_scale_max):
            self.data_scale_max.setValue(0)

        with blocked_signals(self.use_data_clipping_min):
            self.use_data_clipping_min.setChecked(False)

        with blocked_signals(self.use_data_clipping_max):
            self.use_data_clipping_max.setChecked(False)

        with blocked_signals(self.use_data_scale_min):
            self.use_data_scale_min.setChecked(False)

        with blocked_signals(self.use_data_scale_max):
            self.use_data_scale_max.setChecked(False)

        self.info_file_location.setText('-')
        self.info_maximum.setText('-')
        self.info_minimum.setText('-')
        self.info_shape.setText('-')

    def _get_mask(self, data_info, map_name):
        current_model = self._controller.get_model()
        plot_config = current_model.get_config()

        def get_map_attr():
            if map_name in plot_config.map_plot_options:
                value = getattr(plot_config.map_plot_options[map_name], 'mask_name')
                if value is not None:
                    return value
            return plot_config.mask_name

        mask_name = get_map_attr()

        if mask_name is not None:
            return data_info.get_map_data(mask_name)
        return None

    def use(self, map_name):
        """Load the settings of the given map"""
        self._current_map = map_name

        current_model = self._controller.get_model()

        try:
            map_info = current_model.get_config().map_plot_options[map_name]
        except KeyError:
            map_info = SingleMapConfig()

        data_info = current_model.get_data()
        vmin, vmax = data_info.get_single_map_info(map_name).min_max(mask=self._get_mask(data_info, map_name))

        with blocked_signals(self.map_title):
            self.map_title.setText(map_info.title if map_info.title else '')

        with blocked_signals(self.data_colorbar_label):
            self.data_colorbar_label.setText(map_info.colorbar_label if map_info.colorbar_label else '')

        with blocked_signals(self.colormap):
            if map_info.colormap is None:
                self.colormap.setCurrentIndex(0)
            else:
                self.colormap.setCurrentText(map_info.colormap)

        with blocked_signals(self.data_clipping_min):
            self.data_clipping_min.setValue(map_info.clipping.vmin)

        with blocked_signals(self.data_clipping_max):
            self.data_clipping_max.setValue(map_info.clipping.vmax)

        with blocked_signals(self.data_scale_min):
            self.data_scale_min.setValue(map_info.scale.vmin)

        with blocked_signals(self.data_scale_max):
            self.data_scale_max.setValue(map_info.scale.vmax)

        with blocked_signals(self.data_set_use_scale):
            self.data_set_use_scale.setChecked(map_info.scale.use_min or map_info.scale.use_max)

        with blocked_signals(self.data_set_use_clipping):
            self.data_set_use_clipping.setChecked(map_info.clipping.use_min or map_info.clipping.use_max)

        with blocked_signals(self.use_data_clipping_min):
            self.use_data_clipping_min.setChecked(map_info.clipping.use_min)

        with blocked_signals(self.use_data_clipping_max):
            self.use_data_clipping_max.setChecked(map_info.clipping.use_max)

        with blocked_signals(self.use_data_scale_min):
            self.use_data_scale_min.setChecked(map_info.scale.use_min)

        with blocked_signals(self.use_data_scale_max):
            self.use_data_scale_max.setChecked(map_info.scale.use_max)

        map_filename = data_info.get_file_path(map_name)
        if map_filename:
            self.info_file_location.setText(split_long_path_elements(map_filename, 25))

        self.info_maximum.setText(str(vmax))
        self.info_minimum.setText(str(vmin))
        self.info_shape.setText(str(data_info.get_single_map_info(map_name).shape))

    def _get_current_map_config(self):
        current_model = self._controller.get_model()
        current_config = current_model.get_config()
        current_map_config = current_config.map_plot_options.get(self._current_map, SingleMapConfig())
        return current_map_config

    @pyqtSlot(str)
    def _update_map_title(self, string):
        if self._current_map:
            if string == '':
                string = None
            self._controller.apply_action(SetMapTitle(self._current_map, string))

    @pyqtSlot(str)
    def _update_colorbar_label(self, string):
        if self._current_map:
            if string == '':
                string = None
            self._controller.apply_action(SetMapColorbarLabel(self._current_map, string))

    @pyqtSlot(int)
    def _update_colormap(self, index):
        if self._current_map:
            if index == 0:
                self._controller.apply_action(SetMapColormap(self._current_map, None))
            else:
                self._controller.apply_action(SetMapColormap(self._current_map, self.colormap.itemText(index)))

    @pyqtSlot(float)
    def _update_scale_min(self, value):
        if self._current_map:
            current_scale = self._get_current_map_config().scale
            kwargs = dict(vmin=value)

            if current_scale.use_max and current_scale.use_min and value > current_scale.vmax:
                kwargs.update(dict(use_min=False))
                self._auto_enable_scale_min = True
            elif self._auto_enable_scale_min and value < current_scale.vmax:
                kwargs.update(dict(use_min=True))
                self._auto_enable_scale_min = False

            new_scale = current_scale.get_updated(**kwargs)
            self._controller.apply_action(SetMapScale(self._current_map, new_scale))

    @pyqtSlot(float)
    def _update_scale_max(self, value):
        if self._current_map:
            current_scale = self._get_current_map_config().scale
            kwargs = dict(vmax=value)

            if current_scale.use_max and current_scale.use_min and value < current_scale.vmin:
                kwargs.update(dict(use_max=False))
                self._auto_enable_scale_max = True
            elif self._auto_enable_scale_max and value > current_scale.vmin:
                kwargs.update(dict(use_max=True))
                self._auto_enable_scale_max = False

            new_scale = current_scale.get_updated(**kwargs)
            self._controller.apply_action(SetMapScale(self._current_map, new_scale))

    @pyqtSlot(int)
    def _set_use_scale(self, check_state):
        use_scale = check_state == 2
        if self._current_map:
            current_scale = self._get_current_map_config().scale
            if use_scale and current_scale.vmax < current_scale.vmin:
                new_scale = current_scale.get_updated(use_min=use_scale, use_max=use_scale, vmax=current_scale.vmin)
            else:
                new_scale = current_scale.get_updated(use_min=use_scale, use_max=use_scale)
            self._controller.apply_action(SetMapScale(self._current_map, new_scale))

    @pyqtSlot(int)
    def _set_use_data_scale_min(self, check_state):
        use_scale = check_state == 2
        if self._current_map:
            if use_scale and self._get_current_map_config().scale.use_max:
                self._set_use_scale(True)
            else:
                new_scale = self._get_current_map_config().scale.get_updated(use_min=use_scale)
                self._controller.apply_action(SetMapScale(self._current_map, new_scale))

    @pyqtSlot(int)
    def _set_use_data_scale_max(self, check_state):
        use_scale = check_state == 2
        if self._current_map:
            if use_scale and self._get_current_map_config().scale.use_min:
                self._set_use_scale(True)
            else:
                new_scale = self._get_current_map_config().scale.get_updated(use_max=use_scale)
                self._controller.apply_action(SetMapScale(self._current_map, new_scale))

    @pyqtSlot(float)
    def _update_clipping_min(self, value):
        if self._current_map:
            current_clipping = self._get_current_map_config().clipping
            kwargs = dict(vmin=value)

            if current_clipping.use_max and current_clipping.use_min and value > current_clipping.vmax:
                kwargs.update(dict(use_min=False))
                self._auto_enable_scale_min = True
            elif self._auto_enable_scale_min and value < current_clipping.vmax:
                kwargs.update(dict(use_min=True))
                self._auto_enable_scale_min = False

            new_clipping = current_clipping.get_updated(**kwargs)
            self._controller.apply_action(SetMapClipping(self._current_map, new_clipping))

    @pyqtSlot(float)
    def _update_clipping_max(self, value):
        if self._current_map:
            current_clipping = self._get_current_map_config().clipping
            kwargs = dict(vmax=value)

            if current_clipping.use_max and current_clipping.use_min and value < current_clipping.vmin:
                kwargs.update(dict(use_max=False))
                self._auto_enable_scale_max = True
            elif self._auto_enable_scale_max and value > current_clipping.vmin:
                kwargs.update(dict(use_max=True))
                self._auto_enable_scale_max = False

            new_clipping = current_clipping.get_updated(**kwargs)
            self._controller.apply_action(SetMapClipping(self._current_map, new_clipping))

    @pyqtSlot(int)
    def _set_use_clipping(self, check_state):
        use_clipping = check_state == 2
        if self._current_map:
            current_clipping = self._get_current_map_config().clipping
            if use_clipping and current_clipping.vmax < current_clipping.vmin:
                new_clipping = current_clipping.get_updated(use_min=use_clipping, use_max=use_clipping,
                                                            vmax=current_clipping.vmin)
            else:
                new_clipping = current_clipping.get_updated(use_min=use_clipping, use_max=use_clipping)
            self._controller.apply_action(SetMapClipping(self._current_map, new_clipping))

    @pyqtSlot(int)
    def _set_use_data_clipping_min(self, check_state):
        use_clipping = check_state == 2
        if self._current_map:
            if use_clipping and self._get_current_map_config().clipping.use_max:
                self._set_use_clipping(True)
            else:
                new_clipping = self._get_current_map_config().clipping.get_updated(use_min=use_clipping)
                self._controller.apply_action(SetMapClipping(self._current_map, new_clipping))

    @pyqtSlot(int)
    def _set_use_data_clipping_max(self, check_state):
        use_clipping = check_state == 2
        if self._current_map:
            if use_clipping and self._get_current_map_config().clipping.use_min:
                self._set_use_clipping(True)
            else:
                new_clipping = self._get_current_map_config().clipping.get_updated(use_max=use_clipping)
                self._controller.apply_action(SetMapClipping(self._current_map, new_clipping))
