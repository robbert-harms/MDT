import matplotlib
# Make sure that we are using QT5
from PyQt5.QtWidgets import QVBoxLayout
import mdt
from mdt.visualization import MapsVisualizer
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from mdt.gui.maps_visualizer.design.ui_MainWindow import Ui_MapsVisualizer


class MainWindow(QMainWindow, Ui_MapsVisualizer):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.plotting_frame = MatplotlibPlotting(self)
        self.plotLayout.addWidget(self.plotting_frame)

        self.testButton.clicked.connect(self._test)
        self._test_button_pressed_times = 0

    def _test(self):
        print('test button pressed')

        if self._test_button_pressed_times == 0:
            self.plotting_frame.set_slice_index(20)
        else:
            self.plotting_frame.set_dimension(21)

        self._test_button_pressed_times += 1


class ImagesInfo():

    def __init__(self):
        """A container for basic information about the images we are viewing."""

    def get_max_volume(self):
        pass

    def get_max_dimension(self):
        pass

    def get_max_slice_index(self, dimension):
        pass


class PlottingFrame(object):

    def __init__(self):
        super(PlottingFrame, self).__init__()

    def set_dimension(self, dimension):
        """Set the dimension of the plots to the given dimension.

        This only accepts the values 0, 1 and 2 being the three dimensions we visualize. To visualize
        a different volume use the function set_volume_index.

        Args:
            dimension (int): the dimension to set the plots to
        """

    def set_slice_index(self, slice_index):
        """Set the slice index in the current dimension to this value.

        Args:
            slice_index (int): the new slice index
        """


class MatplotlibPlotting(PlottingFrame, QWidget):

    def __init__(self, parent=None):
        super(MatplotlibPlotting, self).__init__()

        names = mdt.results_preselection_names('/media/robbert/01bbb411-36d7-466c-b8f9-ec690d605355/bin/dti_test/output/brain_mask/BallStick/')
        maps = mdt.load_volume_maps('/media/robbert/01bbb411-36d7-466c-b8f9-ec690d605355/bin/dti_test/output/brain_mask/BallStick/', map_names=names)

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


def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()
