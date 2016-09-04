import matplotlib
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget

from mdt.gui.maps_visualizer.actions import SetMapTitle, SetMapColormap, SetMapScale, SetMapClipping
from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig, ValidatedSingleMapConfig
from mdt.gui.maps_visualizer.design.ui_MapSpecificOptions import Ui_MapSpecificOptions
from mdt.gui.maps_visualizer.design.ui_TabMapSpecific import Ui_TabMapSpecific
from mdt.gui.utils import blocked_signals
from mdt.visualization.maps.base import DataInfo, Scale, Clipping

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
        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self.selectedMap.currentIndexChanged.connect(
            lambda ind: self._update_map_specifics(self.selectedMap.itemData(ind, Qt.UserRole)))

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        pass

    @pyqtSlot(ValidatedMapPlotConfig)
    def set_new_config(self, config):
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
            self.map_specific_tab.set_current_map(map_name)


class MapSpecificOptions(QWidget, Ui_MapSpecificOptions):

    def __init__(self, controller, parent=None):
        super(MapSpecificOptions, self).__init__(parent)
        self.setupUi(self)
        self._controller = controller
        self._current_map = None
        self.colormap.addItems(['<disable>'] + sorted(matplotlib.cm.datad))
        self.map_title.textChanged.connect(self._update_map_title)
        self.colormap.currentIndexChanged.connect(self._update_colormap)
        self.data_clipping_min.valueChanged.connect(self._update_clipping_min)
        self.data_clipping_max.valueChanged.connect(self._update_clipping_max)
        self.data_scale_min.valueChanged.connect(self._update_scale_min)
        self.data_scale_max.valueChanged.connect(self._update_scale_max)
        self.data_set_use_scale.stateChanged.connect(self._set_use_scale)
        self.data_set_use_clipping.stateChanged.connect(self._set_use_clipping)

    def reset(self):
        """Set all the values to their defaults"""
        self._current_map = None
        self.colormap.setCurrentText('hot')

        with blocked_signals(self.map_title):
            self.map_title.setText('')

        self.data_clipping_min.setValue(0)
        self.data_clipping_max.setValue(0)
        self.data_scale_min.setValue(0)
        self.data_scale_max.setValue(0)

        self.info_file_location.setText('-')
        self.info_maximum.setText('-')
        self.info_minimum.setText('-')

    def set_current_map(self, map_name):
        """Load the settings of the given map"""
        self._current_map = map_name

        try:
            map_info = self._controller.get_config().map_plot_options[map_name]
        except KeyError:
            map_info = ValidatedSingleMapConfig()

        data_info = self._controller.get_data()
        vmin = data_info.maps[map_name].min()
        vmax = data_info.maps[map_name].max()

        with blocked_signals(self.map_title):
            self.map_title.setText(map_info.title if map_info.title else '')

        with blocked_signals(self.colormap):
            if map_info.colormap is None:
                self.colormap.setCurrentIndex(0)
            else:
                self.colormap.setCurrentText(map_info.colormap)

        with blocked_signals(self.data_clipping_min):
            self.data_clipping_min.setValue(map_info.clipping.vmin if map_info.clipping.vmin is not None else vmin)

        with blocked_signals(self.data_clipping_max):
            self.data_clipping_max.setValue(map_info.clipping.vmax if map_info.clipping.vmax is not None else vmax)

        with blocked_signals(self.data_scale_min):
            self.data_scale_min.setValue(map_info.scale.vmin if map_info.scale.vmin is not None else vmin)

        with blocked_signals(self.data_scale_max):
            self.data_scale_max.setValue(map_info.scale.vmax if map_info.scale.vmax is not None else vmax)

        with blocked_signals(self.data_set_use_scale):
            self.data_set_use_scale.setChecked(map_info.scale != Scale())

        with blocked_signals(self.data_set_use_clipping):
            self.data_set_use_clipping.setChecked(map_info.clipping != Clipping())

        map_filename = data_info.get_file_name(map_name)
        if map_filename:
            self.info_file_location.setText(map_filename)

        self.info_maximum.setText(str(vmax))
        self.info_minimum.setText(str(vmin))

    def _get_current_map_config(self):
        current_config = self._controller.get_config()
        current_map_config = current_config.map_plot_options.get(self._current_map, ValidatedSingleMapConfig())
        return current_map_config

    @pyqtSlot(str)
    def _update_map_title(self, string):
        if self._current_map:
            if string == '':
                string = None
            self._controller.apply_action(SetMapTitle(self._current_map, string))

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
            new_scale = Scale(vmin=value, vmax=current_scale.vmax)
            self._controller.apply_action(SetMapScale(self._current_map, new_scale))

    @pyqtSlot(float)
    def _update_scale_max(self, value):
        if self._current_map:
            current_scale = self._get_current_map_config().scale
            new_scale = Scale(vmin=current_scale.vmin, vmax=value)
            self._controller.apply_action(SetMapScale(self._current_map, new_scale))

    @pyqtSlot(bool)
    def _set_use_scale(self, use_scale):
        if self._current_map:
            if not use_scale:
                self._controller.apply_action(SetMapScale(self._current_map, Scale()))

    @pyqtSlot(float)
    def _update_clipping_min(self, value):
        if self._current_map:
            current_clipping = self._get_current_map_config().clipping
            new_clipping = Clipping(vmin=value, vmax=current_clipping.vmax)
            self._controller.apply_action(SetMapClipping(self._current_map, new_clipping))

    @pyqtSlot(float)
    def _update_clipping_max(self, value):
        if self._current_map:
            current_clipping = self._get_current_map_config().clipping
            new_clipping = Clipping(vmin=current_clipping.vmin, vmax=value)
            self._controller.apply_action(SetMapClipping(self._current_map, new_clipping))

    @pyqtSlot(bool)
    def _set_use_clipping(self, use_clipping):
        if self._current_map:
            if not use_clipping:
                self._controller.apply_action(SetMapClipping(self._current_map, Clipping()))
