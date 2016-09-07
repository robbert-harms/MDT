import matplotlib
import yaml
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMainWindow

from mdt.gui.maps_visualizer.actions import NewConfigAction
from mdt.gui.maps_visualizer.config_tabs.tab_general import TabGeneral
from mdt.gui.maps_visualizer.config_tabs.tab_map_specific import TabMapSpecific
from mdt.gui.maps_visualizer.config_tabs.tab_textual import TabTextual
from mdt.gui.maps_visualizer.design.ui_export_dialog import Ui_ExportImageDialog

matplotlib.use('Qt5Agg')

import mdt
from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig, Controller
from mdt.visualization.maps.base import DataInfo
from mdt.gui.maps_visualizer.renderers.matplotlib_renderer import MatplotlibPlotting
from mdt.gui.model_fit.design.ui_about_dialog import Ui_AboutDialog
from mdt.gui.utils import center_window, DirectoryImageWatcher, QtManager
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

        self.tab_specific = TabMapSpecific(controller, self)
        self.mapSpecificTabPosition.addWidget(self.tab_specific)

        self.tab_textual = TabTextual(controller, self)
        self.textInfoTabPosition.addWidget(self.tab_textual)

        self.auto_render_on.setChecked(True)
        self.auto_render_buttons.buttonClicked.connect(self._set_auto_rendering)
        self.manual_render.clicked.connect(lambda: self.plotting_frame.redraw())

        self.actionAbout.triggered.connect(lambda: AboutDialog(self).exec_())
        self.actionOpen_directory.triggered.connect(self._open_new_directory)
        self.actionExport.triggered.connect(lambda: ExportImageDialog(self, self.plotting_frame).exec_())
        self.actionBrowse_to_current_folder.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self._controller.get_data().directory)))
        self.actionSave_settings.triggered.connect(lambda: self._save_settings())
        self.actionLoad_settings.triggered.connect(lambda: self._load_settings())

        self._status_dir_label = QLabel()
        self.statusBar().addWidget(self._status_dir_label)
        self.statusBar().setStyleSheet("QStatusBar::item { border: 0px solid black }; ")

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        if data_info.directory:
            self._status_dir_label.setText('Loaded directory: ' + data_info.directory)
            self._directory_watcher.set_directory(data_info.directory)
        else:
            self._status_dir_label.setText('No directory information available.')

        self.actionBrowse_to_current_folder.setDisabled(self._controller.get_data().directory is None)

    @pyqtSlot(ValidatedMapPlotConfig)
    def set_new_config(self, config):
        pass

    def _open_new_directory(self):
        initial_dir = self._controller.get_data().directory
        new_dir = QFileDialog(self).getExistingDirectory(caption='Select a folder', directory=initial_dir)
        if new_dir:
            data = DataInfo.from_dir(new_dir)
            if self._controller.get_data().maps:
                start_gui(data, app_exec=False)
            else:
                self._controller.set_data(data)

    @pyqtSlot(tuple, tuple, dict)
    def _update_viewed_images(self, additions, removals, updates):
        data = DataInfo.from_dir(self._controller.get_data().directory)
        if self._controller.get_data().maps:
            config = self._controller.get_config()
        else:
            config = None
        self._controller.set_data(data, config)

    @pyqtSlot()
    def _set_auto_rendering(self):
        auto_render = self.auto_render_on.isChecked()
        self.plotting_frame.set_auto_rendering(auto_render)
        if auto_render:
            self.plotting_frame.redraw()

    def _save_settings(self):
        """Save the current settings as a text file.

        Args:
            file_name: the filename to write to
        """
        config_file = ['conf (*.conf)', 'All files (*)']
        file_name, used_filter = QFileDialog().getSaveFileName(caption='Select the GUI config file',
                                                               filter=';;'.join(config_file))
        if file_name:
            with open(file_name, 'w') as f:
                f.write(self._controller.get_config().to_yaml())

    def _load_settings(self):
        config_file = ['conf (*.conf)', 'All files (*)']
        file_name, used_filter = QFileDialog().getOpenFileName(caption='Select the GUI config file',
                                                               filter=';;'.join(config_file))
        if file_name:
            with open(file_name, 'r') as f:
                try:
                    self._controller.apply_action(NewConfigAction(ValidatedMapPlotConfig.from_yaml(f.read())))
                except yaml.parser.ParserError:
                    pass
                except yaml.scanner.ScannerError:
                    pass
                except ValueError:
                    pass


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
        elif not isinstance(config, ValidatedMapPlotConfig):
            config = ValidatedMapPlotConfig.from_dict(config.to_dict())

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
        applied = self._apply_config(action.apply(self._data_info, self._current_config))
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
            self._apply_config(action.apply(self._data_info, self._current_config))
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


def start_gui(data=None, config=None, controller=None, app_exec=True, show_maximized=False):
    """Start the GUI with the given data and configuration.

    Args:
        data (DataInfo): the initial set of data
        config (ValidatedMapPlotConfig): the initial configuration
        controller (QtController): the controller to use in the application
        app_exec (boolean): if true we execute the Qt application, set to false to disable.
        show_maximized (true): if we want to show the window in a maximized state

    Returns:
        MapsVisualizerWindow: the generated window
    """
    controller = controller or QtController()

    app = QtManager.get_qt_application_instance()

    main = MapsVisualizerWindow(controller)
    center_window(main)

    if show_maximized:
        main.showMaximized()

    main.show()

    if data:
        controller.set_data(data, config)
    elif config:
        controller.set_config(config)

    QtManager.add_window(main)
    if app_exec:
        QtManager.exec_()

    return main
