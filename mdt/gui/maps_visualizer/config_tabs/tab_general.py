import copy
import os

from PyQt5.QtCore import pyqtSlot, Qt, QPoint, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QWidget, QAbstractItemView, QMenu, QWidgetAction, QLabel

from mdt.gui.maps_visualizer.actions import SetDimension, SetSliceIndex, SetVolumeIndex, SetColormap, SetRotate, \
    SetZoom, SetShowAxis, SetColorBarNmrTicks, SetMapsToShow, SetFont, SetInterpolation, SetFlipud, SetPlotTitle, \
    SetGeneralMask, NewDataAction, SetShowPlotColorbars, SetColorbarLocation, SetShowPlotTitles
from mdt.gui.maps_visualizer.base import DataConfigModel
from mdt.gui.maps_visualizer.design.ui_TabGeneral import Ui_TabGeneral
from mdt.gui.utils import blocked_signals, TimedUpdate, split_long_path_elements
from mdt.visualization.maps.base import Zoom, Point2d, DataInfo, Font, MapPlotConfig

__author__ = 'Robbert Harms'
__date__ = "2016-09-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TabGeneral(QWidget, Ui_TabGeneral):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.model_updated.connect(self.update_model)

        self._previous_model = None
        current_model = self._controller.get_model()

        self.general_display_order.setDragDropMode(QAbstractItemView.InternalMove)
        self.general_display_order.setSelectionMode(QAbstractItemView.SingleSelection)

        self.general_colormap.addItems(current_model.get_config().get_available_colormaps())
        self.general_rotate.addItems(['0', '90', '180', '270'])
        self.general_rotate.setCurrentText(str(current_model.get_config().rotate))

        self.general_DisplayOrder.set_collapse(True)
        self.general_Miscellaneous.set_collapse(True)
        self.general_Zoom.set_collapse(True)
        self.general_Font.set_collapse(True)
        self.general_Colorbar.set_collapse(True)

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
        self.general_map_selection.setContextMenuPolicy(Qt.CustomContextMenu)
        self.general_map_selection.customContextMenuRequested.connect(self.select_maps_context_menu)

        self.general_deselect_all_maps.clicked.connect(self._deleselect_all_maps)
        self.general_invert_map_selection.clicked.connect(self._invert_map_selection)

        self.general_zoom_x_0.valueChanged.connect(self._update_zoom)
        self.general_zoom_x_1.valueChanged.connect(self._update_zoom)
        self.general_zoom_y_0.valueChanged.connect(self._update_zoom)
        self.general_zoom_y_1.valueChanged.connect(self._update_zoom)

        self.plot_title.textEdited.connect(lambda txt: self._controller.apply_action(SetPlotTitle(txt)))

        self.general_zoom_reset.clicked.connect(lambda: self._controller.apply_action(SetZoom(Zoom.no_zoom())))
        self.general_zoom_fit.clicked.connect(self._zoom_fit)

        self.general_display_order.items_reordered.connect(self._reorder_maps)
        self.general_show_axis.clicked.connect(lambda: self._controller.apply_action(
            SetShowAxis(self.general_show_axis.isChecked())))

        self.general_font_family.addItems(Font.font_names())
        self.general_font_family.currentTextChanged.connect(
            lambda v: self._controller.apply_action(SetFont(
                self._controller.get_model().get_config().font.get_updated(family=v))))

        self.general_font_size.valueChanged.connect(
            lambda: self._controller.apply_action(SetFont(
                self._controller.get_model().get_config().font.get_updated(size=self.general_font_size.value()))))

        self.general_interpolation.addItems(current_model.get_config().get_available_interpolations())
        self.general_interpolation.currentTextChanged.connect(
            lambda v: self._controller.apply_action(SetInterpolation(v)))

        self.general_flipud.clicked.connect(lambda: self._controller.apply_action(
            SetFlipud(self.general_flipud.isChecked())))

        self.general_show_plot_titles.clicked.connect(lambda: self._controller.apply_action(
            SetShowPlotTitles(self.general_show_plot_titles.isChecked())))

        self.mask_name.currentIndexChanged.connect(self._update_mask_name)

        self.general_colorbar_nmr_ticks.valueChanged.connect(
            lambda v: self._controller.apply_action(SetColorBarNmrTicks(v)))
        self.general_show_colorbar.clicked.connect(lambda: self._controller.apply_action(
            SetShowPlotColorbars(self.general_show_colorbar.isChecked())))
        self.general_colorbar_location.currentTextChanged.connect(
            lambda v: self._controller.apply_action(SetColorbarLocation(v.lower())))

    @pyqtSlot(DataConfigModel)
    def update_model(self, model):
        data = model.get_data()
        config = model.get_config()

        if not self._previous_model or model.get_data() != self._previous_model.get_data():
            self._previous_model = model
            self._update_data(data)

        self._update_config(data, config)

    def select_maps_context_menu(self, position):
        global_position = self.general_map_selection.mapToGlobal(position)

        def get_header_action(parent, map_name):
            label = QLabel(map_name)

            font = label.font()
            font.setBold(True)
            label.setFont(font)
            label.setStyleSheet('color: black; margin:5px; margin-left: 15px;')

            action = QWidgetAction(parent)
            action.setDisabled(True)
            action.setDefaultWidget(label)

            return action

        if self.general_map_selection.count():
            row = self.general_map_selection.indexAt(position)
            if row:
                element = self.general_map_selection.item(row.row())
                if element:
                    map_name = element.data(Qt.UserRole)
                    file_path = self._controller.get_model().get_data().get_file_path(map_name)

                    menu = QMenu()
                    menu.addAction(get_header_action(menu, map_name))
                    menu.addSeparator()

                    show_in_folder = menu.addAction('&Show in folder', lambda:
                        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(file_path))))
                    if file_path is None:
                        show_in_folder.setEnabled(False)

                    menu.addAction('Use as &mask', lambda: self._controller.apply_action(SetGeneralMask(map_name)))
                    menu.addAction('R&emove', lambda: self._controller.apply_action(
                        NewDataAction(self._controller.get_model().get_data().get_updated(removals=[map_name]))))
                    menu.exec(global_position)

    @pyqtSlot()
    def _reorder_maps(self):
        items = [self.general_display_order.item(ind) for ind in range(self.general_display_order.count())]
        map_names = [item.data(Qt.UserRole) for item in items]
        self._controller.apply_action(SetMapsToShow(map_names))

    @pyqtSlot()
    def _update_maps_to_show(self):
        current_model = self._controller.get_model()
        map_names = copy.copy(current_model.get_config().maps_to_show)

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
        current_model = self._controller.get_model()
        self._controller.apply_action(SetMapsToShow(
            list(set(current_model.get_data().get_map_names()).difference(
                set(current_model.get_config().maps_to_show)))))

    @pyqtSlot()
    def _zoom_fit(self):
        current_model = self._controller.get_model()
        data_info = current_model.get_data()
        config = current_model.get_config()

        def add_padding(bounding_box, max_x, max_y):
            bounding_box[0].x = max(bounding_box[0].x - 1, 0)
            bounding_box[0].y = max(bounding_box[0].y - 1, 0)

            bounding_box[1].y = min(bounding_box[1].y + 2, max_y)
            bounding_box[1].x = min(bounding_box[1].x + 2, max_x)

            return bounding_box

        if config.maps_to_show or len(data_info.get_map_names()):
            bounding_box = data_info.get_bounding_box(config.dimension, config.slice_index,
                                                      config.volume_index, config.rotate, config.maps_to_show)

            max_y = data_info.get_max_y_index(config.dimension, rotate=config.rotate, map_names=config.maps_to_show)
            max_x = data_info.get_max_x_index(config.dimension, rotate=config.rotate, map_names=config.maps_to_show)

            if not config.flipud:
                # Since the renderer plots with a left top coordinate system,
                # we need to flip the y coordinates upside down by default.
                tmp = max_y - bounding_box[0].y
                bounding_box[0].y = max_y - bounding_box[1].y
                bounding_box[1].y = tmp

            bounding_box = add_padding(bounding_box, max_x, max_y)

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

    @pyqtSlot(int)
    def _update_mask_name(self, index):
        if index == 0:
            self._controller.apply_action(SetGeneralMask(None))
        else:
            self._controller.apply_action(SetGeneralMask(self.mask_name.itemText(index)))

    def _update_data(self, data_info):
        sorted_keys = list(sorted(data_info.get_map_names()))

        if len(data_info.get_map_names()):
            self.general_info_nmr_maps.setText(str(len(data_info.get_map_names())))
        else:
            self.general_info_nmr_maps.setText('0')

        with blocked_signals(self.general_map_selection):
            self.general_map_selection.clear()
            self.general_map_selection.addItems(sorted_keys)
            for index, map_name in enumerate(sorted_keys):
                item = self.general_map_selection.item(index)
                item.setData(Qt.UserRole, map_name)

        with blocked_signals(self.mask_name):
            self.mask_name.clear()
            self.mask_name.insertItem(0, '-- None --')
            self.mask_name.insertItems(1, sorted_keys)

    def _update_config(self, data, config):
        map_names = config.maps_to_show

        with blocked_signals(self.general_dimension):
            try:
                max_dimension = data.get_max_dimension(map_names)
                self.general_dimension.setMaximum(max_dimension)
                self.maximumDimension.setText(str(max_dimension))
            except ValueError:
                self.general_dimension.setMaximum(0)
                self.maximumDimension.setText(str(0))
            self.general_dimension.setValue(config.dimension)

        with blocked_signals(self.general_slice_index):
            try:
                max_slice = data.get_max_slice_index(config.dimension, map_names)
                self.general_slice_index.setMaximum(max_slice)
                self.maximumIndex.setText(str(max_slice))
            except ValueError:
                self.general_slice_index.setMaximum(0)
                self.maximumIndex.setText(str(0))
            self.general_slice_index.setValue(config.slice_index)

        with blocked_signals(self.general_volume_index):
            try:
                max_volume = data.get_max_volume_index(map_names)
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
                    index = list(sorted(data.get_map_names())).index(map_name)
                    item = self.general_map_selection.item(index)
                    item.setData(Qt.DisplayRole, map_name + ' (' + map_config.title + ')')

            self.general_map_selection.blockSignals(True)
            for index, map_name in enumerate(list(sorted(data.get_map_names()))):
                item = self.general_map_selection.item(index)
                if item:
                    item.setSelected(map_name in map_names)
            self.general_map_selection.blockSignals(False)

        try:
            max_x = data.get_max_x_index(config.dimension, config.rotate, map_names)
            max_y = data.get_max_y_index(config.dimension, config.rotate, map_names)

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

        with blocked_signals(self.plot_title):
            self.plot_title.setText(config.title)

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
            self.general_colorbar_nmr_ticks.setValue(config.colorbar_settings.get_preferred('nmr_ticks'))

        with blocked_signals(self.general_show_colorbar):
            self.general_show_colorbar.setChecked(config.colorbar_settings.get_preferred('visible'))

        with blocked_signals(self.general_colorbar_location):
            self.general_colorbar_location.setCurrentText(config.colorbar_settings.get_preferred('location').title())

        with blocked_signals(self.general_show_plot_titles):
            self.general_show_plot_titles.setChecked(config.show_titles)

        with blocked_signals(self.general_font_family):
            self.general_font_family.setCurrentText(config.font.family)

        with blocked_signals(self.general_font_size):
            self.general_font_size.setValue(config.font.size)

        with blocked_signals(self.general_interpolation):
            self.general_interpolation.setCurrentText(config.interpolation)

        with blocked_signals(self.general_flipud):
            self.general_flipud.setChecked(config.flipud)

        with blocked_signals(self.mask_name):
            if config.mask_name and config.mask_name in data.get_map_names():
                for ind in range(self.mask_name.count()):
                    if self.mask_name.itemText(ind) == config.mask_name:
                        self.mask_name.setCurrentIndex(ind)
                        break
            else:
                self.mask_name.setCurrentIndex(0)
