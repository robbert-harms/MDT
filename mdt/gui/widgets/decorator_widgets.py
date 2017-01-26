from PyQt5.QtWidgets import QDoubleSpinBox

__author__ = 'Robbert Harms'
__date__ = "2017-01-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class QDoubleSpinBoxDotSeparator(QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super(QDoubleSpinBoxDotSeparator, self).__init__(*args, **kwargs)

    def valueFromText(self, text):
        return float(text)

    def textFromValue(self, value):
        return str(value)
