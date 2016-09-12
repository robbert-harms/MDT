import copy
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QAbstractItemView

from mdt.gui.maps_visualizer.actions import SetDimension, SetSliceIndex, SetVolumeIndex, SetColormap, SetRotate, \
    SetZoom, SetShowAxis, SetColorBarNmrTicks, SetMapsToShow, SetFont, SetInterpolation, SetFlipud
from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig
from mdt.gui.maps_visualizer.design.ui_TabGeneral import Ui_TabGeneral
from mdt.gui.utils import blocked_signals, TimedUpdate
from mdt.visualization.maps.base import Zoom, Point, DataInfo, Font

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

        self.general_colormap.addItems(self._controller.get_config().get_available_colormaps())
        self.general_rotate.addItems(['0', '90', '180', '270'])
        self.general_rotate.setCurrentText(str(self._controller.get_config().rotate))

        self.general_DisplayOrder.set_collapse(True)
        self.general_Miscellaneous.set_collapse(True)
        self.general_Zoom.set_collapse(True)
        self.general_Font.set_collapse(True)

        self.general_dimension.valueChanged.connect(lambda v: self._controller.apply_action(SetDimension(v)))
        self.general_slice_index.valueChanged.connect(lambda v: self._controller.apply_action(SetSliceIndex(v)))
        self.general_volume_index.valueChanged.connect(lambda v: self._controller.apply_action(SetVolumeIndex(v)))
        self.general_colormap.currentIndexChanged.connect(
            lambda i: self._controller.apply_action(SetColormap(self.general_colormap.itemText(i))))
        self.general_rotate.currentIndexChanged.connect(
            lambda i: self._controller.apply_action(SetRotate(int(self.general_rotate.itemText(i)))))

        self._map_selection_timer = TimedUpdate(self._update_maps_to_show)
        self.general_map_selection.itemSelectionChanged.connect(
            lambda: self._map_selection_timer.add_delayed_callback(500))

        self.general_deselect_all_maps.clicked.connect(self._deleselect_all_maps)
        self.general_invert_map_selection.clicked.connect(self._invert_map_selection)

        self.general_zoom_x_0.valueChanged.connect(self._update_zoom)
        self.general_zoom_x_1.valueChanged.connect(self._update_zoom)
        self.general_zoom_y_0.valueChanged.connect(self._update_zoom)
        self.general_zoom_y_1.valueChanged.connect(self._update_zoom)

        self.general_zoom_reset.clicked.connect(lambda: self._controller.apply_action(SetZoom(Zoom.no_zoom())))
        self.general_zoom_fit.clicked.connect(self._zoom_fit)

        self.general_display_order.items_reordered.connect(self._reorder_maps)
        self.general_show_axis.clicked.connect(lambda: self._controller.apply_action(
            SetShowAxis(self.general_show_axis.isChecked())))
        self.general_colorbar_nmr_ticks.valueChanged.connect(
            lambda v: self._controller.apply_action(SetColorBarNmrTicks(v)))

        self.general_font_family.addItems(Font.font_names())
        self.general_font_family.currentTextChanged.connect(
            lambda v: self._controller.apply_action(SetFont(self._controller.get_config().font.get_updated(family=v))))

        self.general_font_size.valueChanged.connect(
            lambda: self._controller.apply_action(
                SetFont(self._controller.get_config().font.get_updated(size=self.general_font_size.value()))))

        self.general_interpolation.addItems(self._controller.get_config().get_available_interpolations())
        self.general_interpolation.currentTextChanged.connect(
            lambda v: self._controller.apply_action(SetInterpolation(v)))

        self.general_flipud.clicked.connect(lambda: self._controller.apply_action(
            SetFlipud(self.general_flipud.isChecked())))

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        if data_info.directory:
            self.general_info_directory.setText(data_info.directory)
        else:
            self.general_info_directory.setText('-')

        if len(data_info.maps):
            self.general_info_nmr_maps.setText(str(len(data_info.maps)))
        else:
            self.general_info_nmr_maps.setText('0')

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
            max_y = data_info.get_max_y(config.dimension, config.rotate, map_names)

            with blocked_signals(self.general_zoom_x_0, self.general_zoom_x_1,
                                 self.general_zoom_y_0, self.general_zoom_y_1):
                self.general_zoom_x_0.setMaximum(max_x)
                self.general_zoom_x_0.setValue(config.zoom.p0.x)

                self.general_zoom_x_1.setMaximum(max_x)
                self.general_zoom_x_1.setMinimum(config.zoom.p0.x)
                self.general_zoom_x_1.setValue(config.zoom.p1.x)

                self.general_zoom_y_0.setMaximum(max_y)
                self.general_zoom_y_0.setValue(config.zoom.p0.y)

                self.general_zoom_y_1.setMaximum(max_y)
                self.general_zoom_y_1.setMinimum(config.zoom.p0.y)
                self.general_zoom_y_1.setValue(config.zoom.p1.y)

                if config.zoom.p0.x == 0 and config.zoom.p1.x == 0:
                    self.general_zoom_x_1.setValue(max_x)

                if config.zoom.p0.y == 0 and config.zoom.p1.y == 0:
                    self.general_zoom_y_1.setValue(max_y)
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

        with blocked_signals(self.general_font_family):
            self.general_font_family.setCurrentText(config.font.family)

        with blocked_signals(self.general_font_size):
            self.general_font_size.setValue(config.font.size)

        with blocked_signals(self.general_interpolation):
            self.general_interpolation.setCurrentText(config.interpolation)

        with blocked_signals(self.general_flipud):
            self.general_flipud.setChecked(config.flipud)

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

    @pyqtSlot()
    def _zoom_fit(self):
        data_info = self._controller.get_data()
        config = self._controller.get_config()

        if config.maps_to_show or len(data_info.maps):
            bounding_box = data_info.get_bounding_box(config.dimension, config.slice_index,
                                                      config.volume_index, config.rotate, config.maps_to_show)

            max_y = data_info.get_max_y(config.dimension, rotate=config.rotate, map_names=config.maps_to_show)
            max_x = data_info.get_max_x(config.dimension, rotate=config.rotate, map_names=config.maps_to_show)

            if not config.flipud:
                # Since the renderer plots with a left top coordinate system,
                # we need to flip the y coordinates upside down by default.
                tmp = max_y - bounding_box[0].y
                bounding_box[0].y = max_y - bounding_box[1].y
                bounding_box[1].y = tmp

            if bounding_box[0].x > 0:
                bounding_box[0].x -= 1
            if bounding_box[0].y > 0:
                bounding_box[0].y -= 1

            bounding_box[1].y = min(bounding_box[1].y + 2, max_y)
            bounding_box[1].x = min(bounding_box[1].x + 2, max_x)

            self._controller.apply_action(SetZoom(Zoom(*bounding_box)))

    @pyqtSlot()
    def _update_zoom(self):
        np0x, np0y = self.general_zoom_x_0.value(), self.general_zoom_y_0.value()
        np1x, np1y = self.general_zoom_x_1.value(), self.general_zoom_y_1.value()

        if np0x > np1x:
            np1x = np0x
        if np0y > np1y:
            np1y = np0y

        self._controller.apply_action(SetZoom(Zoom.from_coords(np0x, np0y, np1x, np1y)))

    @staticmethod
    def _insert_alphabetically(new_item, item_list):
        for ind, item in enumerate(item_list):
            if item > new_item:
                item_list.insert(ind, new_item)
                return
        item_list.append(new_item)
