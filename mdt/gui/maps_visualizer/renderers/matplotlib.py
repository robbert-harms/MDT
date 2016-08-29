from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import mdt
from mdt.gui.maps_visualizer.base import PlottingFrame, DataInfo, GeneralConfiguration
from mdt.visualization import MapsVisualizer


class MatplotlibPlotting(PlottingFrame, QWidget):

    def __init__(self, controller, parent=None):
        super(MatplotlibPlotting, self).__init__()
        self._controller = controller

        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self._data_info = controller.get_data()

        self.vis = MapsVisualizer(self._data_info.maps)
        self.vis.show(in_qt=True, show_sliders=False)

        self.canvas = FigureCanvas(self.vis._figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # set the layout
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setParent(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        self._data_info = data_info
        self._redraw()

    @pyqtSlot(GeneralConfiguration)
    def set_new_config(self, configuration):
        self._redraw()

    def _redraw(self):
        width = self.width()
        height = self.height()

        config = self._controller.get_config()
        maps_to_show = [map_name for map_name in config.maps_to_show if map_name in self._data_info.maps]

        self.vis = MapsVisualizer(self._data_info.maps)
        self.vis.show(in_qt=True, show_sliders=False, maps_to_show=maps_to_show,
                      rotate_images=config.rotate)

        self.canvas.figure = self.vis._figure

        self.canvas.resize(width, height-1)
        self.canvas.resize(width, height)
