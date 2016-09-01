import matplotlib
import time

from PyQt5.QtCore import QTimer

matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from mdt.gui.maps_visualizer.base import PlottingFrame, DataInfo, DisplayConfiguration
from mdt.visualization import MapsVisualizer, PlotConfig


class MatplotlibPlotting(PlottingFrame, QWidget):

    def __init__(self, controller, parent=None):
        super(MatplotlibPlotting, self).__init__(controller)

        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self.figure = Figure()
        self.visualizer = MapsVisualizer(self._controller.get_data().maps, self.figure)
        self.visualizer.render(PlotConfig(**self._controller.get_config().to_dict()))

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

        self._timer = QTimer()
        self._timer.timeout.connect(self._timer_event)
        self._timer.timeout.connect(self._timer.stop)

        # todo mouse event handling
        # self.vis._figure.canvas.mpl_connect('motion_notify_event', self._mouse_event)
    #
    # def _mouse_event(self, event):
    #     print(event)

    def export_image(self, filename, width, height, dpi=100):
        width_inch = width / dpi
        height_inch = height / dpi

        figure = Figure(figsize=(width_inch, height_inch), dpi=dpi)
        visualizer = MapsVisualizer(self._controller.get_data().maps, figure)
        FigureCanvas(figure)

        visualizer.to_file(filename, PlotConfig(**self._controller.get_config().to_dict()), dpi=dpi)

    @pyqtSlot()
    def _timer_event(self):
        self._redraw()

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        self.visualizer = MapsVisualizer(data_info.maps, self.figure)
        self._timer.start(300)

    @pyqtSlot(DisplayConfiguration)
    def set_new_config(self, configuration):
        self._timer.start(300)

    def _redraw(self):
        width = self.width()
        height = self.height()

        self.figure.clf()
        self.visualizer.render(PlotConfig(**self._controller.get_config().to_dict()))

        self.canvas.resize(width, height - 1)
        self.canvas.resize(width, height)

