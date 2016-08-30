import matplotlib
matplotlib.use('Qt5Agg')

import copy
import yaml
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog
import mdt
from mdt.gui.maps_visualizer.actions import SetDimension, SetZoom, SetSliceIndex, SetMapsToShow, SetMapTitle, \
    SetMapClipping, FromDictAction, SetVolumeIndex, SetColormap, SetRotate
from mdt.gui.maps_visualizer.base import GeneralConfiguration, Controller, DataInfo, MapSpecificConfiguration
from mdt.gui.maps_visualizer.renderers.matplotlib_renderer import MatplotlibPlotting
from mdt.gui.model_fit.design.ui_about_dialog import Ui_AboutDialog
from mdt.gui.utils import center_window, QApplicationSingleton
import sys
from PyQt5.QtWidgets import QMainWindow
from mdt.gui.maps_visualizer.design.ui_MainWindow import Ui_MapsVisualizer


class MapsVisualizerWindow(QMainWindow, Ui_MapsVisualizer):

    def __init__(self, controller, parent=None):
        super(MapsVisualizerWindow, self).__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self.plotting_frame = MatplotlibPlotting(controller, parent=parent)
        self.plotLayout.addWidget(self.plotting_frame)

        self.general_colormap.addItems(sorted(matplotlib.cm.datad))
        self.general_rotate.addItems(['0', '90', '180', '270'])

        self.general_DisplayOrder.set_collapse(True)
        self.general_Miscellaneous.set_collapse(True)

        self.textConfigEdit.textChanged.connect(self._config_from_string)
        self.general_dimension.valueChanged.connect(lambda v: self._controller.apply_action(SetDimension(v)))
        self.general_slice_index.valueChanged.connect(lambda v: self._controller.apply_action(SetSliceIndex(v)))
        self.general_volume_index.valueChanged.connect(lambda v: self._controller.apply_action(SetVolumeIndex(v)))
        self.general_colormap.currentIndexChanged.connect(
            lambda i: self._controller.apply_action(SetColormap(self.general_colormap.itemText(i))))
        self.general_rotate.currentIndexChanged.connect(
            lambda i: self._controller.apply_action(SetRotate(int(self.general_rotate.itemText(i)))))
        self.general_map_selection.itemSelectionChanged.connect(self._update_maps_to_show)

        self.actionAbout.triggered.connect(lambda: AboutDialog(self).exec_())

        self._flags = {'updating_config_from_string': False}

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        self.general_map_selection.clear()
        self.general_map_selection.addItems(data_info.sorted_keys)
        for index, map_name in enumerate(data_info.sorted_keys):
            item = self.general_map_selection.item(index)
            item.setData(Qt.UserRole, map_name)

        if data_info.directory:
            self.statusBar().showMessage('Loaded directory: ' + data_info.directory)
        else:
            self.statusBar().showMessage('No directory information available.')
        self.set_new_config(self._controller.get_config())

    @pyqtSlot(GeneralConfiguration)
    def set_new_config(self, config):
        data_info = self._controller.get_data()
        map_names = config.maps_to_show

        self.general_dimension.setValue(config.dimension)
        self.general_slice_index.setValue(config.slice_index)
        self.general_volume_index.setValue(config.volume_index)

        self.general_dimension.setMaximum(data_info.get_max_dimension(map_names))
        self.general_slice_index.setMaximum(data_info.get_max_slice_index(config.dimension, map_names))
        self.general_volume_index.setMaximum(data_info.get_max_volume_index(map_names))

        self.general_colormap.setCurrentText(config.colormap)
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
                item.setSelected(map_name in map_names)
            self.general_map_selection.blockSignals(False)

        if not self._flags['updating_config_from_string']:
            yaml_string = yaml.dump(config.to_dict())
            self.textConfigEdit.setPlainText(yaml_string)

    @pyqtSlot()
    def _config_from_string(self):
        self._flags['updating_config_from_string'] = True
        text = self.textConfigEdit.toPlainText()
        text = text.replace('\t', ' '*4)
        try:
            info_dict = yaml.load(text)
            self._controller.apply_action(FromDictAction(info_dict))
        except yaml.parser.ParserError:
            pass
        except yaml.scanner.ScannerError:
            pass
        finally:
            self._flags['updating_config_from_string'] = False

    @pyqtSlot()
    def _update_maps_to_show(self):
        map_names = copy.copy(self._controller.get_config().maps_to_show)

        for item in [self.general_map_selection.item(ind) for ind in range(self.general_map_selection.count())]:
            map_name = item.data(Qt.UserRole)

            if item.isSelected():
                print(map_name)
                if map_name not in map_names:
                    self._insert_alphabetically(map_name, map_names)
            else:
                if map_name in map_names:
                    map_names.remove(map_name)

        self._controller.apply_action(SetMapsToShow(map_names))

    @staticmethod
    def _insert_alphabetically(new_item, item_list):
        for ind, item in enumerate(item_list):
            if item < new_item:
                item_list.insert(ind, new_item)
                return
        item_list.append(new_item)


class AboutDialog(Ui_AboutDialog, QDialog):

    def __init__(self, parent):
        super(AboutDialog, self).__init__(parent)
        self.setupUi(self)
        self.contentLabel.setText(self.contentLabel.text().replace('{version}', mdt.__version__))


class QtController(Controller, QObject):

    new_data = pyqtSignal(DataInfo)
    new_config = pyqtSignal(GeneralConfiguration)

    def __init__(self):
        super(QtController, self).__init__()
        self._data_info = DataInfo({})
        self._actions_history = []
        self._redoable_actions = []
        self._current_config = GeneralConfiguration()

    def set_data(self, data_info, config=None):
        self._data_info = data_info
        self._actions_history = []
        self._redoable_actions = []

        if not config:
            config = GeneralConfiguration()
            config.maps_to_show = mdt.results_preselection_names(data_info.maps)
            config.slice_index = None

        self._apply_config(config)
        self.new_data.emit(data_info)

    def get_data(self):
        return self._data_info

    def set_config(self, general_config):
        self._apply_config(general_config)

    def get_config(self):
        return self._current_config

    def apply_action(self, action):
        print('add_action')
        applied = self._apply_config(action.apply(self._current_config))
        if applied:
            self._actions_history.append(action)
            self._redoable_actions = []

    def undo(self):
        print('undo')
        if len(self._actions_history):
            action = self._actions_history.pop()
            self._apply_config(action.unapply())
            self._redoable_actions.append(action)

    def redo(self):
        print('redo')
        if len(self._redoable_actions):
            action = self._redoable_actions.pop()
            self._apply_config(action.apply(self._current_config))
            self._actions_history.append(action)

    def _apply_config(self, new_config):
        """Apply the current configuration.

        Args:
            new_config (GeneralConfiguration): the new configuration to apply

        Returns:
            bool: if the configuration was applied or not. If the difference with the current configuration
                and the old one is None, False is returned. Else True is returned.
        """
        validated_config = new_config.validate(self._data_info)
        difference = self._current_config.get_difference(validated_config)

        print('apply_config', difference)

        if difference:
            print('applying')
            self._current_config = validated_config
            self.new_config.emit(validated_config)
            return True
        return False


def start_gui(data=None, config=None):
    controller = QtController()
    app = QApplicationSingleton.get_instance()
    main = MapsVisualizerWindow(controller)
    center_window(main)
    main.show()

    if data:
        controller.set_data(data, config)
    elif config:
        controller.set_config(config)

    sys.exit(app.exec_())


if __name__ == '__main__':
    #
    # # # data = DataInfo.from_dir('/home/robbert/phd-data/dti_test_ballstick_results/')
    # # data = DataInfo.from_dir('/home/robbert/phd-data/dti_test/output/brain_mask/BallStick/')
    data = DataInfo.from_dir('/home/robbert/phd-data/dti_test/output/4Ddwi_b1000_mask_2_25/BallStick/')
    config = GeneralConfiguration()
    config.maps_to_show = ['S0.s0', 'BIC']
    config.zoom['x_0'] = 20
    config.zoom['y_0'] = 10
    config.zoom['x_1'] = 80
    config.zoom['y_1'] = 80

    config.map_plot_options.update({'S0.s0': MapSpecificConfiguration(title='S0 test')})
    config.map_plot_options.update({'BIC': MapSpecificConfiguration(title='S0 test',
                                                                    scale={'max': 200, 'min': 0})})
    config.slice_index = None

    start_gui(data, config)




#
# data = DataInfo.from_dir('/home/robbert/phd-data/dti_test/output/brain_mask/BallStick/')
#
# print(data.get_max_volume_index(['Stick.vec0', 'S0.s0']))
#
# config = GeneralConfiguration()
# config.maps_to_show = ['S0.s0', 'BIC']
# config.dimension = 2
# # new_config = SetMapsToShow(['S0.s0', 'B']).apply(config)
# validated = config.validate(data)
# diff = config.get_difference(validated)
#
# print(diff)
# #
