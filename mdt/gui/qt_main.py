import sys
from PyQt5 import QtGui
from pkg_resources import resource_filename
from PyQt5.QtWidgets import QMainWindow, QApplication
from mdt.gui.qt.design.ui_main_window import Ui_MainWindow

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MDTGUISingleModel(QMainWindow, Ui_MainWindow):

    def __init__(self, initial_directory=None):
        super(MDTGUISingleModel, self).__init__()
        self.setupUi(self)
        self.initial_directory = initial_directory


def start_single_model_gui(initial_directory=None):
    app = QApplication([])
    form = MDTGUISingleModel(initial_directory)
    #
    # icon = QtGui.QIcon()
    # icon.addPixmap(QtGui.QPixmap(resource_filename('mdt', 'data/logo.gif')), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    # form.setWindowIcon(icon)

    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    start_single_model_gui()
