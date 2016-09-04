import copy

import matplotlib
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QAbstractItemView

from mdt.gui.maps_visualizer.actions import SetDimension, SetSliceIndex, SetVolumeIndex, SetColormap, SetRotate, \
    SetZoom, SetShowAxis, SetColorBarNmrTicks, SetMapsToShow
from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig
from mdt.gui.maps_visualizer.design.ui_TabGeneral import Ui_TabGeneral
from mdt.gui.utils import blocked_signals
from mdt.visualization.maps.base import Zoom, Point, DataInfo

__author__ = 'Robbert Harms'
__date__ = "2016-09-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TabGeneral(QWidget, Ui_TabGeneral):

    def __init__(self, controller, parent=None):
        super(TabGeneral, self).__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self.general_display_order.setDragDropMode(QAbstractItemView.InternalMove)
        self.general_display_order.setSelectionMode(QAbstractItemView.SingleSelection)

        self.general_colormap.addItems(sorted(matplotlib.cm.datad))
        self.general_rotate.addItems(['0', '90', '180', '270'])
        self.general_rotate.setCurrentText(str(self._controller.get_config().rotate))

        self.general_DisplayOrder.set_collapse(True)
        self.general_Miscellaneous.set_collapse(True)

        self.general_dimension.valueChanged.connect(lambda v: self._controller.apply_action(SetDimension(v)))
        self.general_slice_index.valueChanged.connect(lambda v: self._controller.apply_action(SetSliceIndex(v)))
        self.general_volume_index.valueChanged.connect(lambda v: self._controller.apply_action(SetVolumeIndex(v)))
        self.general_colormap.currentIndexChanged.connect(
            lambda i: self._controller.apply_action(SetColormap(self.general_colormap.itemText(i))))
        self.general_rotate.currentIndexChanged.connect(
            lambda i: self._controller.apply_action(SetRotate(int(self.general_rotate.itemText(i)))))
        self.general_map_selection.itemSelectionChanged.connect(self._update_maps_to_show)

        self.general_deselect_all_maps.clicked.connect(self._deleselect_all_maps)
        self.general_invert_map_selection.clicked.connect(self._invert_map_selection)

        self.general_zoom_x_0.valueChanged.connect(lambda v: self._controller.apply_action(SetZoom(
            Zoom(Point(v, self._controller.get_config().zoom.p0.y),
                 self._controller.get_config().zoom.p1))))
        self.general_zoom_x_1.valueChanged.connect(lambda v: self._controller.apply_action(SetZoom(
            Zoom(self._controller.get_config().zoom.p0,
                 Point(v, self._controller.get_config().zoom.p1.y)))))
        self.general_zoom_y_0.valueChanged.connect(lambda v: self._controller.apply_action(SetZoom(
            Zoom(Point(self._controller.get_config().zoom.p0.x, v),
                 self._controller.get_config().zoom.p1))))
        self.general_zoom_y_1.valueChanged.connect(lambda v: self._controller.apply_action(SetZoom(
            Zoom(self._controller.get_config().zoom.p0,
                 Point(self._controller.get_config().zoom.p1.x, v)))))

        self.general_display_order.items_reordered.connect(self._reorder_maps)
        self.general_show_axis.clicked.connect(lambda: self._controller.apply_action(
            SetShowAxis(self.general_show_axis.isChecked())))
        self.general_colorbar_nmr_ticks.valueChanged.connect(
            lambda v: self._controller.apply_action(SetColorBarNmrTicks(v)))

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        with blocked_signals(self.general_map_selection):
            self.general_map_selection.clear()
            self.general_map_selection.addItems(data_info.sorted_keys)
            for index, map_name in enumerate(data_info.sorted_keys):
                item = self.general_map_selection.item(index)
                item.setData(Qt.UserRole, map_name)

    @pyqtSlot(ValidatedMapPlotConfig)
    def set_new_config(self, config):
        data_info = self._controller.get_data()
        map_names = config.maps_to_show

        with blocked_signals(self.general_dimension):
            try:
                max_dimension = data_info.get_max_dimension(map_names)
                self.general_dimension.setMaximum(max_dimension)
                self.maximumDimension.setText(str(max_dimension))
            except ValueError:
                self.general_dimension.setMaximum(0)
                self.maximumDimension.setText(str(0))
            self.general_dimension.setValue(config.dimension)

        with blocked_signals(self.general_slice_index):
            try:
                max_slice = data_info.get_max_slice_index(config.dimension, map_names)
                self.general_slice_index.setMaximum(max_slice)
                self.maximumIndex.setText(str(max_slice))
            except ValueError:
                self.general_slice_index.setMaximum(0)
                self.maximumIndex.setText(str(0))
            self.general_slice_index.setValue(config.slice_index)

        with blocked_signals(self.general_volume_index):
            try:
                max_volume = data_info.get_max_volume_index(map_names)
                self.general_volume_index.setMaximum(max_volume)
                self.maximumVolume.setText(str(max_volume))
            except ValueError:
                self.general_volume_index.setMaximum(0)
                self.maximumVolume.setText(str(0))
            self.general_volume_index.setValue(config.volume_index)

        with blocked_signals(self.general_colormap):
            self.general_colormap.setCurrentText(config.colormap)

        with blocked_signals(self.general_rotate):
            self.general_rotate.setCurrentText(str(config.rotate))

        if self.general_map_selection.count():
            for map_name, map_config in config.map_plot_options.items():
                if map_config.title:
                    index = data_info.sorted_keys.index(map_name)
                    item = self.general_map_selection.item(index)
                    item.setData(Qt.DisplayRole, map_name + ' (' + map_config.title + ')')

            self.general_map_selection.blockSignals(True)
            for index, map_name in enumerate(data_info.sorted_keys):
                item = self.general_map_selection.item(index)
                if item:
                    item.setSelected(map_name in map_names)
            self.general_map_selection.blockSignals(False)

        try:
            max_x = data_info.get_max_x(config.dimension, config.rotate, map_names)
            with blocked_signals(self.general_zoom_x_0):
                self.general_zoom_x_0.setMaximum(max_x)
                self.general_zoom_x_0.setValue(config.zoom.p0.x)

            with blocked_signals(self.general_zoom_x_1):
                self.general_zoom_x_1.setMaximum(max_x)
                self.general_zoom_x_1.setMinimum(config.zoom.p0.x)
                self.general_zoom_x_1.setValue(config.zoom.p1.x)

            max_y = data_info.get_max_y(config.dimension, config.rotate, map_names)
            with blocked_signals(self.general_zoom_y_0):
                self.general_zoom_y_0.setMaximum(max_y)
                self.general_zoom_y_0.setValue(config.zoom.p0.y)

            with blocked_signals(self.general_zoom_y_1):
                self.general_zoom_y_1.setMaximum(max_y)
                self.general_zoom_y_1.setMinimum(config.zoom.p0.y)
                self.general_zoom_y_1.setValue(config.zoom.p1.y)
        except ValueError:
            pass

        with blocked_signals(self.general_display_order):
            self.general_display_order.clear()
            self.general_display_order.addItems(map_names)

            for index, map_name in enumerate(map_names):
                item = self.general_display_order.item(index)
                item.setData(Qt.UserRole, map_name)

                if map_name in config.map_plot_options and config.map_plot_options[map_name].title:
                    title = config.map_plot_options[map_name].title
                    item.setData(Qt.DisplayRole, map_name + ' (' + title + ')')

        with blocked_signals(self.general_show_axis):
            self.general_show_axis.setChecked(config.show_axis)

        with blocked_signals(self.general_colorbar_nmr_ticks):
            self.general_colorbar_nmr_ticks.setValue(config.colorbar_nmr_ticks)

    @pyqtSlot()
    def _reorder_maps(self):
        items = [self.general_display_order.item(ind) for ind in range(self.general_display_order.count())]
        map_names = [item.data(Qt.UserRole) for item in items]
        self._controller.apply_action(SetMapsToShow(map_names))

    @pyqtSlot()
    def _update_maps_to_show(self):
        map_names = copy.copy(self._controller.get_config().maps_to_show)

        for item in [self.general_map_selection.item(ind) for ind in range(self.general_map_selection.count())]:
            map_name = item.data(Qt.UserRole)

            if item.isSelected():
                if map_name not in map_names:
                    self._insert_alphabetically(map_name, map_names)
            else:
                if map_name in map_names:
                    map_names.remove(map_name)

        self._controller.apply_action(SetMapsToShow(map_names))

    @pyqtSlot()
    def _deleselect_all_maps(self):
        self._controller.apply_action(SetMapsToShow([]))

    @pyqtSlot()
    def _invert_map_selection(self):
        self._controller.apply_action(SetMapsToShow(
            set(self._controller.get_data().maps.keys()).difference(set(self._controller.get_config().maps_to_show))))

    @staticmethod
    def _insert_alphabetically(new_item, item_list):
        for ind, item in enumerate(item_list):
            if item > new_item:
                item_list.insert(ind, new_item)
                return
        item_list.append(new_item)
