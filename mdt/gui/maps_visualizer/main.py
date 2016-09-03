from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtWidgets import QApplication
import copy
import yaml
import yaml.parser
import yaml.scanner
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMainWindow

import matplotlib
from PyQt5.QtWidgets import QWidget

from mdt.gui.maps_visualizer.design.ui_TabGeneral import Ui_TabGeneral
from mdt.gui.maps_visualizer.design.ui_export_dialog import Ui_ExportImageDialog

matplotlib.use('Qt5Agg')

import mdt
from mdt.gui.maps_visualizer.actions import SetDimension, SetZoom, SetSliceIndex, SetMapsToShow, \
    NewConfigAction, SetVolumeIndex, SetColormap, SetRotate, SetFont, SetShowAxis, SetColorBarNmrTicks
from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig, Controller, ValidatedSingleMapConfig
from mdt.visualization.maps.base import DataInfo, Zoom, Point, Clipping, Scale
from mdt.gui.maps_visualizer.renderers.matplotlib_renderer import MatplotlibPlotting
from mdt.gui.model_fit.design.ui_about_dialog import Ui_AboutDialog
from mdt.gui.utils import center_window, blocked_signals, DirectoryImageWatcher
from mdt.gui.maps_visualizer.design.ui_MainWindow import Ui_MapsVisualizer


class MapsVisualizerWindow(QMainWindow, Ui_MapsVisualizer):

    def __init__(self, controller, parent=None):
        super(MapsVisualizerWindow, self).__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self._directory_watcher = DirectoryImageWatcher()
        self._directory_watcher.image_updates.connect(self._update_viewed_images)

        self.plotting_frame = MatplotlibPlotting(controller, parent=parent)
        self.plotLayout.addWidget(self.plotting_frame)

        self.tab_general = TabGeneral(controller, self)
        self.generalTabPosition.addWidget(self.tab_general)

        self.textConfigEdit.new_config.connect(self._config_from_string)

        self.actionAbout.triggered.connect(lambda: AboutDialog(self).exec_())
        self.actionOpen_directory.triggered.connect(self._open_new_directory)
        self.actionExport.triggered.connect(lambda: ExportImageDialog(self, self.plotting_frame).exec_())
        self.actionBrowse_to_current_folder.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self._controller.get_data().directory)))

        self._flags = {'updating_config_from_string': False}

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        if data_info.directory:
            status_label = QLabel('Loaded directory: ' + data_info.directory)
            self._directory_watcher.set_directory(data_info.directory)
        else:
            status_label = QLabel('No directory information available.')

        self.actionBrowse_to_current_folder.setDisabled(self._controller.get_data().directory is None)

        self.statusBar().addWidget(status_label)
        self.statusBar().setStyleSheet("QStatusBar::item { border: 0px solid black }; ")

    @pyqtSlot(ValidatedMapPlotConfig)
    def set_new_config(self, config):
        with blocked_signals(self.textConfigEdit):
            if not self._flags['updating_config_from_string']:
                self.textConfigEdit.setPlainText(config.to_yaml())

    @pyqtSlot(str)
    def _config_from_string(self, text):
        self._flags['updating_config_from_string'] = True
        text = text.replace('\t', ' '*4)
        try:
            self._controller.apply_action(NewConfigAction(ValidatedMapPlotConfig.from_yaml(text)))
        except yaml.parser.ParserError:
            pass
        except yaml.scanner.ScannerError:
            pass
        except ValueError:
            pass
        finally:
            self._flags['updating_config_from_string'] = False

    def _open_new_directory(self):
        initial_dir = self._controller.get_data().directory
        new_dir = QFileDialog(self).getExistingDirectory(caption='Select a folder', directory=initial_dir)
        if new_dir:
            controller = QtController()
            main = MapsVisualizerWindow(controller)
            center_window(main)
            main.show()
            controller.set_data(DataInfo.from_dir(new_dir))

    @pyqtSlot(tuple, tuple, dict)
    def _update_viewed_images(self, additions, removals, updates):
        data = DataInfo.from_dir(self._controller.get_data().directory)
        if self._controller.get_data().maps:
            config = self._controller.get_config()
        else:
            config = None
        self._controller.set_data(data, config)


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
            items = [self.general_display_order.item(ind) for ind in range(self.general_display_order.count())]
            current_order = [item.data(Qt.UserRole) for item in items]

            if current_order != map_names:
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

    @staticmethod
    def _insert_alphabetically(new_item, item_list):
        for ind, item in enumerate(item_list):
            if item > new_item:
                item_list.insert(ind, new_item)
                return
        item_list.append(new_item)


class ExportImageDialog(Ui_ExportImageDialog, QDialog):

    def __init__(self, parent, plotting_frame):
        super(ExportImageDialog, self).__init__(parent)
        self.setupUi(self)
        self._plotting_frame = plotting_frame
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self._export_image)
        self.outputFile_box.textChanged.connect(self._update_ok_button)
        self.outputFile_chooser.clicked.connect(lambda: self._select_file())

    @pyqtSlot()
    def _update_ok_button(self):
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(self.outputFile_box.text() != '')

    def _select_file(self):
        graphical_image_filters = ['png (*.png)', 'All files (*)']

        open_file, used_filter = QFileDialog().getSaveFileName(caption='Select the output file',
                                                               filter=';;'.join(graphical_image_filters))

        if open_file:
            self.outputFile_box.setText(open_file)
            self._update_ok_button()

    def _export_image(self):
        self._plotting_frame.export_image(self.outputFile_box.text(), self.width_box.value(), self.height_box.value(),
                                          dpi=self.dpi_box.value())


class AboutDialog(Ui_AboutDialog, QDialog):

    def __init__(self, parent):
        super(AboutDialog, self).__init__(parent)
        self.setupUi(self)
        self.contentLabel.setText(self.contentLabel.text().replace('{version}', mdt.__version__))


class QtController(Controller, QObject):

    new_data = pyqtSignal(DataInfo)
    new_config = pyqtSignal(ValidatedMapPlotConfig)

    def __init__(self):
        super(QtController, self).__init__()
        self._data_info = DataInfo({})
        self._actions_history = []
        self._redoable_actions = []
        self._current_config = ValidatedMapPlotConfig()

    def set_data(self, data_info, config=None):
        self._data_info = data_info
        self._actions_history = []
        self._redoable_actions = []

        if not config:
            config = ValidatedMapPlotConfig()
            config.maps_to_show = mdt.results_preselection_names(data_info.maps)
            config.slice_index = None

        self._apply_config(config)
        self.new_data.emit(data_info)
        self.new_config.emit(self._current_config)

    def get_data(self):
        return self._data_info

    def set_config(self, general_config):
        applied = self._apply_config(general_config)
        if applied:
            self._actions_history.clear()
            self._redoable_actions.clear()
            self.new_config.emit(self._current_config)

    def get_config(self):
        return self._current_config

    def apply_action(self, action):
        applied = self._apply_config(action.apply(self._current_config))
        if applied:
            self._actions_history.append(action)
            self._redoable_actions = []
            self.new_config.emit(self._current_config)

    def undo(self):
        if len(self._actions_history):
            action = self._actions_history.pop()
            self._apply_config(action.unapply())
            self._redoable_actions.append(action)
            self.new_config.emit(self._current_config)

    def redo(self):
        if len(self._redoable_actions):
            action = self._redoable_actions.pop()
            self._apply_config(action.apply(self._current_config))
            self._actions_history.append(action)
            self.new_config.emit(self._current_config)

    def _apply_config(self, new_config):
        """Apply the current configuration.

        Args:
            new_config (ValidatedMapPlotConfig): the new configuration to apply

        Returns:
            bool: if the configuration was applied or not. If the difference with the current configuration
                and the old one is None, False is returned. Else True is returned.
        """
        validated_config = new_config.validate(self._data_info)
        if self._current_config != validated_config:
            self._current_config = validated_config
            return True
        return False


def start_gui(data=None, config=None):
    controller = QtController()

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    main = MapsVisualizerWindow(controller)
    center_window(main)
    main.show()

    if data:
        controller.set_data(data, config)
    elif config:
        controller.set_config(config)

    app.exec_()


if __name__ == '__main__':

    # data = DataInfo.from_dir('/home/robbert/phd-data/dti_test_ballstick_results/')
    data = DataInfo.from_dir('/home/robbert/phd-data/dti_test/output/brain_mask/BallStick/')
    data = DataInfo.from_dir('/home/robbert/phd-data/dti_test/output/4Ddwi_b1000_mask_2_25/BallStick/')
    config = ValidatedMapPlotConfig()
    config.maps_to_show = ['S0.s0', 'BIC']
    config.zoom = Zoom(Point(20, 10), Point(80, 80))
    config.map_plot_options.update({'S0.s0': ValidatedSingleMapConfig(title='S0 test')})
    config.map_plot_options.update({'BIC': ValidatedSingleMapConfig(title='BIC test',
                                                                    clipping=Clipping(vmax=150),
                                                                    scale=Scale(vmin=0, vmax=200))})
    config.slice_index = None

    # data = DataInfo.from_dir('/tmp/test')
    # config = None

    start_gui(data, config)




# #
# data = DataInfo.from_dir('/home/robbert/phd-data/dti_test/output/brain_mask/BallStick/')
# #
# # print(data.get_max_volume_index(['Stick.vec0', 'S0.s0']))
# #
# config = ValidatedMapPlotConfig()
# config.maps_to_show = ['S0.s0', 'BIC']
# config.dimension = 2
# new_config = SetMapsToShow(['S0.s0', 'B']).apply(config)
# validated = config.validate(data)
# # diff = config.get_difference(validated)
#
# # print(config != new_config)
