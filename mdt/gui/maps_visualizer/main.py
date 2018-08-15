import copy
import os
import signal
from textwrap import dedent

import matplotlib
matplotlib.use('Qt5Agg')

import yaml
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QMessageBox

from mdt.gui.maps_visualizer.actions import NewConfigAction, SetMapsToShow, NewDataAction
from mdt.gui.maps_visualizer.config_tabs.tab_general import TabGeneral
from mdt.gui.maps_visualizer.config_tabs.tab_map_specific import TabMapSpecific
from mdt.gui.maps_visualizer.config_tabs.tab_textual import TabTextual
from mdt.gui.maps_visualizer.design.ui_save_image_dialog import Ui_SaveImageDialog
from mdt.lib.nifti import is_nifti_file

import mdt
from mdt.gui.maps_visualizer.base import DataConfigModel, \
    QtController
from mdt.gui.maps_visualizer.renderers.base import PlottingFrameInfoViewer
from mdt.visualization.maps.base import DataInfo, SimpleDataInfo, MapPlotConfig
from mdt.visualization.maps.utils import get_shortest_unique_names
from mdt.gui.maps_visualizer.renderers.matplotlib_renderer import MatplotlibPlotting
from mdt.gui.model_fit.design.ui_about_dialog import Ui_AboutDialog
from mdt.gui.utils import center_window, QtManager, get_script_file_header_text, image_files_filters, \
    enable_pyqt_exception_hook
from mdt.gui.maps_visualizer.design.ui_MainWindow import Ui_MapsVisualizer


class MapsVisualizerWindow(QMainWindow, Ui_MapsVisualizer):

    def __init__(self, controller, parent=None):
        """Instantiate the maps GUI

        Args:
            controller (mdt.gui.maps_visualizer.base.Controller): the controller to use for updating the views
        """
        super().__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.model_updated.connect(self.update_model)

        self.setAcceptDrops(True)

        self._coordinates_label = QLabel()

        self.statusBar().addPermanentWidget(self._coordinates_label)
        self.statusBar().setStyleSheet("QStatusBar::item { border: 0px solid black }; ")

        self.plotting_info_to_statusbar = PlottingFrameInfoToStatusBar(self._controller, self._coordinates_label)
        self.plotting_frame = MatplotlibPlotting(controller, parent=parent,
                                                 plotting_info_viewer=self.plotting_info_to_statusbar)
        self.plotLayout.addWidget(self.plotting_frame)

        self.tab_general = TabGeneral(controller, self)
        self.generalTabPosition.addWidget(self.tab_general)

        self.tab_specific = TabMapSpecific(controller, self)
        self.mapSpecificTabPosition.addWidget(self.tab_specific)

        self.tab_textual = TabTextual(controller, self)
        self.textInfoTabPosition.addWidget(self.tab_textual)

        self.auto_rendering.setChecked(True)
        self.auto_rendering.stateChanged.connect(self._set_auto_rendering)
        self.manual_render.clicked.connect(lambda: self.plotting_frame.redraw())

        self.actionNew_window.triggered.connect(lambda: start_gui(app_exec=False))
        self.actionAbout.triggered.connect(lambda: AboutDialog(self).exec_())
        self.actionAdd_new_files.triggered.connect(self._add_new_files)
        self.action_Clear.triggered.connect(self._remove_files)
        self.actionSaveImage.triggered.connect(lambda: ExportImageDialog(self, self.plotting_frame,
                                                                         self._controller).exec_())
        self.actionSave_settings.triggered.connect(lambda: self._save_settings())
        self.actionLoad_settings.triggered.connect(lambda: self._load_settings())

        self.undo_config.setDisabled(not self._controller.has_undo())
        self.redo_config.setDisabled(not self._controller.has_redo())

        self.undo_config.clicked.connect(lambda: self._controller.undo())
        self.redo_config.clicked.connect(lambda: self._controller.redo())

        self._qdialog_basedir_set = False

    @pyqtSlot(DataConfigModel)
    def update_model(self, model):
        self.undo_config.setDisabled(not self._controller.has_undo())
        self.redo_config.setDisabled(not self._controller.has_redo())
        self._set_qdialog_basedir()

    def _set_qdialog_basedir(self):
        if not self._qdialog_basedir_set:
            data = self._controller.get_model().get_data()
            for map_name, file_path in data.get_file_paths().items():
                if file_path:
                    QFileDialog().setDirectory(file_path)
                    self._qdialog_basedir_set = True
                    return

    def resizeEvent(self, event):
        ExportImageDialog.plot_frame_resized()

    def _add_new_files(self):
        new_files = QFileDialog(self).getOpenFileNames(caption='Nifti files',
                                                       filter=';;'.join(image_files_filters))
        if new_files[0]:
            self._add_new_maps(new_files[0])

    def _add_new_maps(self, paths):
        """Add the given set of file paths to the current visualization.

        This looks at the current set of file paths (from ``self._controller.get_config().get_data()``)
        and the given new set of file paths and creates a new merged dataset with all files.

        Since it is possible that adding new maps leads to naming collisions this function can rename both the
        old and the new maps to better reflect the map names.

        Args:
            paths (list of str): the list of file paths to add to the visualization

        Returns:
            list of str: the display names of the newly added maps
        """
        def get_file_paths(data_info):
            """Get the file paths"""
            paths = []
            for map_name in data_info.get_map_names():
                file_path = data_info.get_file_path(map_name)
                if file_path:
                    paths.append(file_path)
                else:
                    paths.append(map_name)
            return paths

        def get_changes(old_data_info, new_data_info):
            additions = {}
            removals = []
            name_updates = {}

            paths = get_file_paths(old_data_info) + get_file_paths(new_data_info)
            unique_names = get_shortest_unique_names(paths)
            new_map_names = unique_names[len(old_data_info.get_map_names()):]

            for old_name, new_name in zip(old_data_info.get_map_names(), unique_names):
                if old_name != new_name:
                    removals.append(old_name)
                    additions[new_name] = old_data_info.get_single_map_info(old_name)
                    name_updates[old_name] = new_name

            for new_map_name, new_map_rename in zip(new_data_info.get_map_names(), new_map_names):
                additions[new_map_rename] = new_data_info.get_single_map_info(new_map_name)

            return additions, removals, name_updates, new_map_names

        current_model = self._controller.get_model()
        current_data = current_model.get_data()

        adds, rems, name_updates, new_map_names = get_changes(current_data, SimpleDataInfo.from_paths(paths))
        data = current_data.get_updated(adds, removals=rems)
        config = copy.deepcopy(current_model.get_config())

        new_maps_to_show = []
        for map_name in config.maps_to_show:
            if map_name in name_updates:
                new_maps_to_show.append(name_updates[map_name])
            else:
                new_maps_to_show.append(map_name)
        config.maps_to_show = new_maps_to_show

        new_map_plot_options = {}
        for map_name, plot_options in config.map_plot_options.items():
            if map_name in name_updates:
                new_map_plot_options[name_updates[map_name]] = plot_options
            else:
                new_map_plot_options[map_name] = plot_options
        config.map_plot_options = new_map_plot_options

        if not len(current_data.get_map_names()):
            config.slice_index = data.get_max_slice_index(config.dimension) // 2

        self._controller.apply_action(NewDataAction(data, config=config))
        return new_map_names

    def _remove_files(self):
        data = SimpleDataInfo({})
        config = MapPlotConfig()
        self._controller.apply_action(NewDataAction(data, config))

    @pyqtSlot()
    def _set_auto_rendering(self):
        auto_render = self.auto_rendering.isChecked()
        self.plotting_frame.set_auto_rendering(auto_render)
        if auto_render:
            self.plotting_frame.redraw()

    def send_sigint(self, *args):
        self.close()

    def dragEnterEvent(self, event):
        """Function to allow dragging nifti files in the viewer for viewing purpose."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """One or more files where dropped in the GUI, load all the nifti files among them."""
        nifti_paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path) and is_nifti_file(path):
                nifti_paths.append(path)

        additional_maps = self._add_new_maps(nifti_paths)

        map_names = copy.copy(self._controller.get_model().get_config().maps_to_show)
        map_names.extend(additional_maps)

        self._controller.apply_action(SetMapsToShow(map_names))

    def _save_settings(self):
        """Save the current settings as a text file.
        """
        current_model = self._controller.get_model()

        config_file = ['conf (*.conf)', 'All files (*)']
        file_name, used_filter = QFileDialog().getSaveFileName(caption='Select the GUI config file',
                                                               filter=';;'.join(config_file))
        if file_name:
            with open(file_name, 'w') as f:
                f.write(current_model.get_config().to_yaml())

    def _load_settings(self):
        config_file = ['conf (*.conf)', 'All files (*)']
        file_name, used_filter = QFileDialog().getOpenFileName(caption='Select the GUI config file',
                                                               filter=';;'.join(config_file))
        if file_name:
            with open(file_name, 'r') as f:
                try:
                    self._controller.apply_action(NewConfigAction(MapPlotConfig.from_yaml(f.read())))
                except yaml.parser.ParserError:
                    pass
                except yaml.scanner.ScannerError:
                    pass
                except ValueError:
                    pass

    def set_window_title(self, title):
        if title is None:
            self.setWindowTitle('MDT Maps Visualizer')
        else:
            self.setWindowTitle('MDT Maps Visualizer - {}'.format(title))


class PlottingFrameInfoToStatusBar(PlottingFrameInfoViewer):

    def __init__(self, controller, status_bar_label):
        super().__init__()
        self._controller = controller
        self._status_bar_label = status_bar_label

    def set_voxel_info(self, map_name, onscreen_coords, data_index):
        super().set_voxel_info(map_name, onscreen_coords, data_index)

        def format_value(v):
            value_format = '{:.3e}'
            if 1e-3 < v < 1e3:
                value_format = '{:.3f}'
            return value_format.format(v)

        data = self._controller.get_model().get_data()

        if map_name in data.get_map_names():
            value = data.get_map_data(map_name)[tuple(data_index)]
            clipped = value

            config = self._controller.get_model().get_config()
            if map_name in config.map_plot_options:
                clipped = config.map_plot_options[map_name].clipping.apply(value)

            if clipped != value:
                self._status_bar_label.setText("{}, {}, {} ({})".format(
                    onscreen_coords, data_index, format_value(clipped), format_value(value)))
            else:
                self._status_bar_label.setText("{}, {}, {}".format(onscreen_coords, data_index, format_value(value)))

    def clear_voxel_info(self):
        super().clear_voxel_info()
        self._status_bar_label.setText("")


class ExportImageDialog(Ui_SaveImageDialog, QDialog):

    previous_values = {'width': None, 'height': None,
                       'dpi': None, 'output_file': None,
                       'writeScriptsAndConfig': False}

    def __init__(self, parent, plotting_frame, controller):
        super().__init__(parent)
        self._extension_filters = [['png', '(*.png)'], ['svg', '(*.svg)']]
        self.setupUi(self)
        self._plotting_frame = plotting_frame
        self._controller = controller

        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self._export_image)
        self.outputFile_box.textChanged.connect(self._update_ok_button)
        self.outputFile_chooser.clicked.connect(lambda: self._select_file())

        if self.previous_values['width']:
            self.width_box.setValue(self.previous_values['width'])
        else:
            self.width_box.setValue(self._plotting_frame.width())

        if self.previous_values['height']:
            self.height_box.setValue(self.previous_values['height'])
        else:
            self.height_box.setValue(self._plotting_frame.height())

        if self.previous_values['dpi']:
            self.dpi_box.setValue(self.previous_values['dpi'])
        if self.previous_values['output_file']:
            self.outputFile_box.setText(self.previous_values['output_file'])
        if self.previous_values['writeScriptsAndConfig'] is not None:
            self.writeScriptsAndConfig.setChecked(self.previous_values['writeScriptsAndConfig'])

    @staticmethod
    def plot_frame_resized():
        ExportImageDialog.previous_values['width'] = None
        ExportImageDialog.previous_values['height'] = None

    @pyqtSlot()
    def _update_ok_button(self):
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(self.outputFile_box.text() != '')

    def _select_file(self):
        graphical_image_filters = [' '.join(el) for el in self._extension_filters] + ['All files (*)']

        open_file, used_filter = QFileDialog().getSaveFileName(caption='Select the output file',
                                                               filter=';;'.join(graphical_image_filters))

        if not any(open_file.endswith(el[0]) for el in self._extension_filters):
            extension_from_filter = list(filter(lambda v: ' '.join(v) == used_filter, self._extension_filters))
            if extension_from_filter:
                extension = extension_from_filter[0][0]
            else:
                extension = self._extension_filters[0][0]

            open_file += '.{}'.format(extension)

        if open_file:
            self.outputFile_box.setText(open_file)
            self._update_ok_button()

    def _export_image(self):
        output_file = self.outputFile_box.text()
        if not any(output_file.endswith(el[0]) for el in self._extension_filters):
            output_file += '.{}'.format(self._extension_filters[0][0])

        try:
            self._plotting_frame.export_image(output_file, self.width_box.value(), self.height_box.value(),
                                              dpi=self.dpi_box.value())
            self.previous_values['width'] = self.width_box.value()
            self.previous_values['height'] = self.height_box.value()
            self.previous_values['dpi'] = self.dpi_box.value()
            self.previous_values['output_file'] = self.outputFile_box.text()
            self.previous_values['writeScriptsAndConfig'] = self.writeScriptsAndConfig.isChecked()

            if self.writeScriptsAndConfig.isChecked():
                output_basename = os.path.splitext(output_file)[0]
                self._write_config_file(output_basename + '.conf')
                self._write_python_script_file(output_basename + '_script.py', output_basename + '.conf', output_file,
                                             self.width_box.value(), self.height_box.value(), self.dpi_box.value())
                self._write_bash_script_file(output_basename + '_script.sh', output_basename + '.conf', output_file,
                                             self.width_box.value(), self.height_box.value(), self.dpi_box.value())

        except PermissionError as error:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Could not write the file to the given destination.")
            msg.setInformativeText(str(error))
            msg.setWindowTitle("Permission denied")
            msg.exec_()

    def _write_config_file(self, output_basename):
        current_model = self._controller.get_model()
        with open(output_basename, 'w') as f:
            f.write(current_model.get_config().to_yaml())

    def _write_python_script_file(self, script_fname, configuration_fname, output_image_fname, width, height, dpi):
        with open(script_fname, 'w') as f:
            f.write('#!/usr/bin/env python\n')
            f.write(dedent('''
                {header}

                import mdt

                with open({config!r}, 'r') as f:
                    config = f.read()

                mdt.view_maps(
                    {paths},
                    config=config,
                    save_filename={output_name!r},
                    figure_options={{'width': {width}, 'height': {height}, 'dpi': {dpi}}})

            ''').format(header=get_script_file_header_text({'Purpose': 'Generate a results figure'}),
                        paths='[' + ', '.join(['{el!r}'.format(el=el) for el in self._get_file_paths()]) + ']',
                        config=configuration_fname,
                        output_name=output_image_fname,
                        width=width, height=height, dpi=dpi))

    def _write_bash_script_file(self, script_fname, configuration_fname, output_image_fname, width, height, dpi):
        with open(script_fname, 'w') as f:
            f.write('#!/usr/bin/env bash\n')
            f.write(dedent('''
                {header}

                mdt-view-maps \\
                    {paths} \\
                    --config "{config}" \\
                    --to-file "{output_name}" \\
                    --width {width} \\
                    --height {height} \\
                    --dpi {dpi}
            ''').format(header=get_script_file_header_text({'Purpose': 'Generate a results figure'}),
                        paths=' '.join(['{el!r}'.format(el=el) for el in self._get_file_paths()]),
                        config=configuration_fname,
                        output_name=output_image_fname,
                        width=width, height=height, dpi=dpi))

    def _get_file_paths(self):
        data = self._controller.get_model().get_data()
        return list(data.get_file_paths().values())


class AboutDialog(Ui_AboutDialog, QDialog):

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.contentLabel.setText(self.contentLabel.text().replace('{version}', mdt.__version__))


def start_gui(data=None, config=None, controller=None, app_exec=True, show_maximized=False, window_title=None):
    """Start the GUI with the given data and configuration.

    Args:
        data (DataInfo): the initial set of data
        config (MapPlotConfig): the initial configuration
        controller (mdt.gui.maps_visualizer.base.QtController): the controller to use in the application
        app_exec (boolean): if true we execute the Qt application, set to false to disable.
        show_maximized (true): if we want to show the window in a maximized state
        window_title (str): the title of the window

    Returns:
        MapsVisualizerWindow: the generated window
    """
    controller = controller or QtController()

    enable_pyqt_exception_hook()
    app = QtManager.get_qt_application_instance()

    # catches the sigint
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    main = MapsVisualizerWindow(controller)
    main.set_window_title(window_title)
    signal.signal(signal.SIGINT, main.send_sigint)

    center_window(main)

    if show_maximized:
        main.showMaximized()

    main.show()

    if data is None:
        data = SimpleDataInfo({})
    controller.apply_action(NewDataAction(data, config=config), store_in_history=False)

    QtManager.add_window(main)
    if app_exec:
        QtManager.exec_()

    return main
