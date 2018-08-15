import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget

from mdt.gui.utils import TimedUpdate
from mdt.gui.widgets.design.ui_scientific_number_scroller_widget import Ui_ScientificScroller

__author__ = 'Robbert Harms'
__date__ = "2017-01-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ScientificNumberScroller(Ui_ScientificScroller, QWidget):

    valueChanged = pyqtSignal([float], ['QString'])

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.mantissa.valueChanged.connect(self._value_changed_cb)
        self.exponent.valueChanged.connect(self._value_changed_cb)
        self._timer = TimedUpdate(self._signal_new_value)
        self._update_delay = 0

    @pyqtSlot()
    def _value_changed_cb(self):
        return self._timer.add_delayed_callback(self._update_delay)

    def set_update_delay(self, update_delay):
        self._update_delay = update_delay

    @pyqtSlot()
    def _signal_new_value(self):
        self.valueChanged.emit(self.mantissa.value() * 10**self.exponent.value())

    @pyqtSlot(float)
    def setValue(self, value):
        if value == 0:
            self.exponent.setValue(0)
            self.mantissa.setValue(0)
        elif 1e-3 < np.abs(value) < 1e3:
            self.exponent.setValue(0)
            self.mantissa.setValue(value)
        else:
            exponent = int(np.floor(np.log10(np.abs(value))))
            self.exponent.setValue(exponent)
            self.mantissa.setValue(value / 10**exponent)

    def blockSignals(self, block):
        self.mantissa.blockSignals(block)
        return self.exponent.blockSignals(block)
