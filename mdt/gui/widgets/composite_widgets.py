import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget

from mdt.gui.widgets.design.ui_scientific_number_scroller_widget import Ui_ScientificScroller

__author__ = 'Robbert Harms'
__date__ = "2017-01-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ScientificNumberScroller(Ui_ScientificScroller, QWidget):

    valueChanged = pyqtSignal([float], ['QString'])

    def __init__(self, parent):
        super(ScientificNumberScroller, self).__init__(parent)
        self.setupUi(self)
        self.mantissa.valueChanged.connect(self._signal_new_value)
        self.exponent.valueChanged.connect(self._signal_new_value)

    @pyqtSlot()
    def _signal_new_value(self):
        self.valueChanged.emit(self.mantissa.value() * 10**self.exponent.value())

    @pyqtSlot(float)
    def setValue(self, value):
        if value == 0:
            self.exponent.setValue(0)
            self.mantissa.setValue(0)
        else:
            exponent = np.floor(np.log10(np.abs(value))).astype(int)
            self.exponent.setValue(exponent)
            self.mantissa.setValue(value / 10**exponent)

    def blockSignals(self, block):
        self.mantissa.blockSignals(block)
        return self.exponent.blockSignals(block)
