import matplotlib

from mdt.gui.maps_visualizer.actions import SetDimension, SetZoom, SetSliceIndex, SetMapsToShow, SetMapTitle, \
    SetMapClipping
from mdt.gui.maps_visualizer.base import GeneralConfiguration, Controller
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

        self.controller = controller

        self.plotting_frame = MatplotlibPlotting(controller, parent=parent)
        self.plotLayout.addWidget(self.plotting_frame)

        self.general_DisplayOrder.set_collapse(True)
        self.general_Miscellaneous.set_collapse(True)


class QtController(Controller):

    def __init__(self):
        super(QtController, self).__init__()
        self._actions_history = []
        self._redoable_actions = []
        self._current_config = GeneralConfiguration()

    def set_config(self, general_config):
        self._apply_config(general_config)

    def get_config(self):
        return self._current_config

    def add_action(self, action):
        self._actions_history.append(action)
        self._redoable_actions = []
        self._apply_config(action.apply(self._current_config))

    def undo(self):
        if len(self._actions_history):
            action = self._actions_history.pop()
            self._redoable_actions.append(action)
            self._apply_config(action.unapply())

    def redo(self):
        if len(self._redoable_actions):
            action = self._redoable_actions.pop()
            self._actions_history.append(action)
            self._apply_config(action.apply(self._current_config))

    def _apply_config(self, new_config):
        """Apply the current configuration"""
        print(self._current_config.get_difference(new_config))
        self._current_config = new_config


def main():
    controller = QtController()
    app = QApplication(sys.argv)
    main = MainWindow(controller)
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

# controller = QtController()
# controller.add_action(SetDimension(1))
# controller.add_action(SetDimension(2))
# controller.add_action(SetDimension(3))
# controller.undo()
# controller.undo()
# controller.redo()
#
# controller.add_action(SetDimension(4))
# config = controller.get_config()
# config.get_dict()
