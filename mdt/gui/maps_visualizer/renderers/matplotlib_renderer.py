import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')

from PyQt5.QtCore import QTimer

from mdt.visualization.maps.matplotlib_renderer import MapsVisualizer
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from mdt.gui.maps_visualizer.base import PlottingFrame
from mdt.visualization.maps.base import DataInfo, MapPlotConfig


class MatplotlibPlotting(PlottingFrame, QWidget):

    def __init__(self, controller, parent=None, plotting_info_viewer=None):
        super(MatplotlibPlotting, self).__init__(controller, plotting_info_viewer=plotting_info_viewer)

        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self._auto_render = True

        self.figure = Figure(facecolor='#bfbfbf')
        self.visualizer = MapsVisualizer(self._controller.get_data(), self.figure)
        self._axes_data = self.visualizer.render(self._controller.get_config())

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

        self._redraw_timer = QTimer()
        self._redraw_timer.timeout.connect(self._timer_event)
        self._redraw_timer.timeout.connect(self._redraw_timer.stop)

        self._mouse_interaction = _MouseInteraction(self.figure, self._plotting_info_viewer)
        self._mouse_interaction.update_axes_data(self._axes_data)

        self._previous_config = None

        self.setMinimumWidth(100)

    def export_image(self, filename, width, height, dpi=100):
        width_inch = width / dpi
        height_inch = height / dpi

        figure = Figure(figsize=(width_inch, height_inch), dpi=dpi)
        visualizer = MapsVisualizer(self._controller.get_data(), figure)
        FigureCanvas(figure)

        visualizer.to_file(filename, self._controller.get_config(), dpi=dpi)

    def set_auto_rendering(self, auto_render):
        self._auto_render = auto_render

    def redraw(self):
        self._redraw()

    @pyqtSlot()
    def _timer_event(self):
        self._redraw()

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        self.visualizer = MapsVisualizer(data_info, self.figure)
        self._redraw_timer.start(300)

    @pyqtSlot(MapPlotConfig)
    def set_new_config(self, configuration):
        if not self._previous_config or configuration.visible_changes(self._previous_config):
            self._previous_config = configuration
            if self._auto_render:
                self._redraw_timer.start(300)

    def _redraw(self):
        self.figure.clf()

        self._axes_data = self.visualizer.render(self._controller.get_config())
        self._mouse_interaction.update_axes_data(self._axes_data)

        self.figure.canvas.draw()


class _MouseInteraction(object):

    def __init__(self, figure, plotting_info_viewer):
        self.figure = figure
        self.plotting_info_viewer = plotting_info_viewer
        self._axes_data = []
        self.figure.canvas.mpl_connect('button_release_event', self._button_released)
        self.figure.canvas.mpl_connect('motion_notify_event', self._mouse_motion)

    def update_axes_data(self, axes_data):
        """Set the updated axes data. Needs to be called if the axes are updated.

        Args:
            axes_data (list of AxisData): the information about the axes
        """
        self._axes_data = axes_data

    def _button_released(self, event):
        axis_data = self._get_matching_axis_data(event.inaxes)
        if axis_data:
            x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
            index = axis_data.coordinates_to_index(x, y)
            value = axis_data.get_value(index)
            # todo draw info box on the figure
            # print(x, y, index, value)

    def _mouse_motion(self, event):
        axis_data = self._get_matching_axis_data(event.inaxes)
        if axis_data:
            x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
            index = axis_data.coordinates_to_index(x, y)
            value = axis_data.get_value(index)
            self.plotting_info_viewer.set_voxel_info((x, y), tuple(index), float(value))
        else:
            self.plotting_info_viewer.clear_voxel_info()

    def _get_matching_axis_data(self, axis):
        """Get the axis data matching the given axis.

        Args:
            Axis: the matplotlib axis to match

        Returns:
            AxisData: our data container for that axis
        """
        if axis:
            for axes_data in self._axes_data:
                if axes_data.axis == axis:
                    return axes_data
        return None
