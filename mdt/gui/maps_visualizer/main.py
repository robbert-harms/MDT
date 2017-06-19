import copy
import os
import signal
from textwrap import dedent

import matplotlib
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
from mdt.nifti import is_nifti_file, load_nifti
from mdt.utils import split_image_path

matplotlib.use('Qt5Agg')

import mdt
from mdt.gui.maps_visualizer.base import DataConfigModel, \
    QtController
from mdt.gui.maps_visualizer.renderers.base import PlottingFrameInfoViewer
from mdt.visualization.maps.base import DataInfo, SimpleDataInfo, MapPlotConfig, SingleMapInfo
from mdt.gui.maps_visualizer.renderers.matplotlib_renderer import MatplotlibPlotting
from mdt.gui.model_fit.design.ui_about_dialog import Ui_AboutDialog
from mdt.gui.utils import center_window, QtManager, get_script_file_header_text, image_files_filters
from mdt.gui.maps_visualizer.design.ui_MainWindow import Ui_MapsVisualizer


class MapsVisualizerWindow(QMainWindow, Ui_MapsVisualizer):

    def __init__(self, controller, parent=None):
        super(MapsVisualizerWindow, self).__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.model_updated.connect(self.update_model)

        self.setAcceptDrops(True)

        self._coordinates_label = QLabel()

        self.statusBar().addPermanentWidget(self._coordinates_label)
        self.statusBar().setStyleSheet("QStatusBar::item { border: 0px solid black }; ")

        self.plotting_info_to_statusbar = PlottingFrameInfoToStatusBar(self._coordinates_label)
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

    @pyqtSlot(DataConfigModel)
    def update_model(self, model):
        self.undo_config.setDisabled(not self._controller.has_undo())
        self.redo_config.setDisabled(not self._controller.has_redo())

    def _add_new_files(self):
        current_model = self._controller.get_model()

        new_files = QFileDialog(self).getOpenFileNames(caption='Nifti files',
                                                       filter=';;'.join(image_files_filters))
        if new_files[0]:
            additional_maps = {}
            for nifti_path in new_files[0]:
                folder, basename, ext = split_image_path(nifti_path)
                additional_maps.update({basename: SingleMapInfo.from_file(nifti_path)})

            data = current_model.get_data().get_updated(additional_maps)
            config = current_model.get_config()

            if not len(current_model.get_data().get_map_names()):
                config.slice_index = data.get_max_slice_index(config.dimension) // 2

            self._controller.apply_action(NewDataAction(data, config))

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

        additional_maps = {}
        for nifti_path in nifti_paths:
            folder, basename, ext = split_image_path(nifti_path)
            additional_maps.update({basename: SingleMapInfo.from_file(nifti_path)})

        current_model = self._controller.get_model()
        map_names = copy.copy(current_model.get_config().maps_to_show)
        map_names.extend(additional_maps)
        self._controller.apply_action(NewDataAction(current_model.get_data().get_updated(additional_maps)))
        self._controller.apply_action(SetMapsToShow(map_names))

    def _save_settings(self):
        """Save the current settings as a text file.

        Args:
            file_name: the filename to write to
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
        self.setWindowTitle('MDT Maps Visualizer - {}'.format(title))


class PlottingFrameInfoToStatusBar(PlottingFrameInfoViewer):

    def __init__(self, status_bar_label):
        super(PlottingFrameInfoToStatusBar, self).__init__()
        self._status_bar_label = status_bar_label

    def set_voxel_info(self, onscreen_coords, data_index, value):
        super(PlottingFrameInfoToStatusBar, self).set_voxel_info(onscreen_coords, data_index, value)

        value_format = '{:.3e}'
        if 1e-3 < value < 1e3:
            value_format = '{:.3f}'

        self._status_bar_label.setText("{}, {}, {}".format(onscreen_coords, data_index, value_format.format(value)))

    def clear_voxel_info(self):
        super(PlottingFrameInfoToStatusBar, self).clear_voxel_info()
        self._status_bar_label.setText("")


class ExportImageDialog(Ui_SaveImageDialog, QDialog):

    previous_values = {'width': None, 'height': None,
                       'dpi': None, 'output_file': None,
                       'writeScriptsAndConfig': None}

    def __init__(self, parent, plotting_frame, controller):
        super(ExportImageDialog, self).__init__(parent)
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
        if self.previous_values['height']:
            self.height_box.setValue(self.previous_values['height'])
        if self.previous_values['dpi']:
            self.dpi_box.setValue(self.previous_values['dpi'])
        if self.previous_values['output_file']:
            self.outputFile_box.setText(self.previous_values['output_file'])
        if self.previous_values['writeScriptsAndConfig'] is not None:
            self.writeScriptsAndConfig.setChecked(self.previous_values['writeScriptsAndConfig'])

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

                mdt.write_view_maps_figure(
                    {paths},
                    {output_name!r},
                    config=config,
                    width={width},
                    height={height},
                    dpi={dpi})

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
        file_paths = []
        for map_name in data.get_map_names():
            file_paths.append(data.get_file_path(map_name))
        return file_paths


class AboutDialog(Ui_AboutDialog, QDialog):

    def __init__(self, parent):
        super(AboutDialog, self).__init__(parent)
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
