import os

import matplotlib
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget
from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig, ValidatedSingleMapConfig
from mdt.gui.maps_visualizer.design.ui_MapSpecificOptions import Ui_MapSpecificOptions
from mdt.gui.maps_visualizer.design.ui_TabMapSpecific import Ui_TabMapSpecific
from mdt.gui.utils import blocked_signals
from mdt.visualization.maps.base import DataInfo

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
            current_items = [self.selectedMap.itemData(ind, Qt.UserRole) for ind in range(self.selectedMap.count())]

            if current_items != map_names:
                self.selectedMap.clear()
                self.selectedMap.addItems(map_names)

                for index, map_name in enumerate(map_names):
                    self.selectedMap.setItemData(index, map_name, Qt.UserRole)

                    if map_name in config.map_plot_options and config.map_plot_options[map_name].title:
                        title = config.map_plot_options[map_name].title
                        self.selectedMap.setItemData(index, map_name + ' (' + title + ')', Qt.DisplayRole)

            self.selectedMap.setCurrentIndex(0)

        if self.selectedMap.count():
            self._update_map_specifics(self.selectedMap.itemData(0, Qt.UserRole))
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
        self.colormap.addItems(sorted(matplotlib.cm.datad))

    def reset(self):
        """Set all the values to their defaults"""
        self.colormap.setCurrentText('hot')
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

        self.map_title.setText(map_info.title if map_info.title else '')
        self.colormap.setCurrentText(map_info.colormap)

        self.data_clipping_min.setValue(map_info.clipping.vmin if map_info.clipping.vmin is not None else 0)
        self.data_clipping_max.setValue(map_info.clipping.vmax if map_info.clipping.vmax is not None else 0)

        self.data_scale_min.setValue(map_info.scale.vmin if map_info.scale.vmin is not None else 0)
        self.data_scale_max.setValue(map_info.scale.vmax if map_info.scale.vmax is not None else 0)

        map_filename = data_info.get_file_name(map_name)
        if map_filename:
            self.info_file_location.setText(map_filename)

        self.info_maximum.setText(str(data_info.maps[map_name].max()))
        self.info_minimum.setText(str(data_info.maps[map_name].min()))
