from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import mdt
from mdt.gui.maps_visualizer.base import PlottingFrame
from mdt.visualization import MapsVisualizer


class MatplotlibPlotting(PlottingFrame, QWidget):

    def __init__(self, controller, parent=None):
        super(MatplotlibPlotting, self).__init__()
        self.controller = controller

        names = mdt.results_preselection_names('/media/robbert/01bbb411-36d7-466c-b8f9-ec690d605355/bin/dti_test/output/brain_mask/BallStick/')
        maps = mdt.load_volume_maps('/media/robbert/01bbb411-36d7-466c-b8f9-ec690d605355/bin/dti_test/output/brain_mask/BallStick/', map_names=names)
        # maps = {}

        self.vis = MapsVisualizer(maps)
        self.vis.show(in_qt=True, show_sliders=False)

        canvas = FigureCanvas(self.vis._figure)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        canvas.updateGeometry()

        # set the layout
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        self.setLayout(layout)

        self.setParent(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()

        self.vis._figure.canvas.mpl_connect('button_press_event', self._on_press)

    def set_dimension(self, dimension):
        self.vis.set_dimension(dimension)

    def set_slice_index(self, slice_index):
        self.vis.set_slice_ind(slice_index)

    def _on_press(self, event):
        print(event)
