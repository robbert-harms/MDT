from PyQt5.QtWidgets import QDoubleSpinBox

__author__ = 'Robbert Harms'
__date__ = "2017-01-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class QDoubleSpinBoxDotSeparator(QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return 0

    def textFromValue(self, value):
        return str(value)
