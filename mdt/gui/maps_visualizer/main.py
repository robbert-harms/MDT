import copy

import matplotlib
import yaml
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

import mdt
from mdt.gui.maps_visualizer.actions import SetDimension, SetZoom, SetSliceIndex, SetMapsToShow, SetMapTitle, \
    SetMapClipping, FromDictAction
from mdt.gui.maps_visualizer.base import GeneralConfiguration, Controller, DataInfo
from mdt.gui.maps_visualizer.renderers.matplotlib import MatplotlibPlotting

matplotlib.use('Qt5Agg')
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from mdt.gui.maps_visualizer.design.ui_MainWindow import Ui_MapsVisualizer


class MainWindow(QMainWindow, Ui_MapsVisualizer):

    def __init__(self, controller, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)
        self.textConfigEdit.textChanged.connect(self._config_from_string)

        self.plotting_frame = MatplotlibPlotting(controller, parent=parent)
        self.plotLayout.addWidget(self.plotting_frame)

        self.general_DisplayOrder.set_collapse(True)
        self.general_Miscellaneous.set_collapse(True)

        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Z), self, self._controller.undo)
        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Y), self, self._controller.redo)

        self._flags = {'updating_config_from_string': False}

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        if data_info.directory:
            self.statusBar().showMessage('Loaded directory: ' + data_info.directory)
        else:
            self.statusBar().showMessage('No directory information available.')
        self.set_new_config(self._controller.get_config())

    @pyqtSlot(GeneralConfiguration)
    def set_new_config(self, configuration):
        if not self._flags['updating_config_from_string']:
            yaml_string = yaml.dump(configuration.to_dict())
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
        finally:
            self._flags['updating_config_from_string'] = False


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

        if config:
            self._current_config = config
        else:
            self._current_config = GeneralConfiguration()
            self._current_config.maps_to_show = mdt.results_preselection_names(data_info.maps)

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


def main():
    controller = QtController()
    app = QApplication(sys.argv)
    main = MainWindow(controller)
    main.show()

    # # data = DataInfo.from_dir('/home/robbert/phd-data/dti_test_ballstick_results/')
    data = DataInfo.from_dir('/home/robbert/phd-data/dti_test/output/brain_mask/BallStick/')
    config = GeneralConfiguration()
    config.maps_to_show = ['S0.s0', 'BIC']
    controller.set_data(data, config)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()



data = DataInfo.from_dir('/home/robbert/phd-data/dti_test/output/brain_mask/BallStick/')

print(data.get_max_volume_index(['Stick.vec0', 'S0.s0']))

config = GeneralConfiguration()
config.maps_to_show = ['S0.s0', 'BIC']
config.dimension = 2
# new_config = SetMapsToShow(['S0.s0', 'B']).apply(config)
validated = config.validate(data)
diff = config.get_difference(validated)

print(diff)
#
