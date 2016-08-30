import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from mdt.gui.maps_visualizer.base import PlottingFrame, DataInfo, GeneralConfiguration
from mdt.visualization import MapsVisualizer


class MatplotlibPlotting(PlottingFrame, QWidget):

    def __init__(self, controller, parent=None):
        super(MatplotlibPlotting, self).__init__()
        self._controller = controller

        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self.figure = plt.figure()
        self._init_visualizer()

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setParent(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()

        # todo mouse event handling
        # self.vis._figure.canvas.mpl_connect('motion_notify_event', self._mouse_event)
    #
    # def _mouse_event(self, event):
    #     print(event)

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        self._redraw()

    @pyqtSlot(GeneralConfiguration)
    def set_new_config(self, configuration):
        self._redraw()

    def _redraw(self):
        width = self.width()
        height = self.height()

        self._init_visualizer()

        self.canvas.resize(width, height-1)
        self.canvas.resize(width, height)

    def _init_visualizer(self):
        self.figure.clf()
        config = self._controller.get_config()
        vis = MapsVisualizer(self._controller.get_data().maps, self.figure)
        vis.render(**config.to_dict())
