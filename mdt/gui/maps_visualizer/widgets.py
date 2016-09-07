import re

from PyQt5.QtCore import QEvent
from PyQt5.QtCore import QObject
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtWidgets import QPlainTextEdit


class CollapsablePanel(QFrame):

    def __init__(self, parent=None):
        super(CollapsablePanel, self).__init__(parent)

    def toggle(self):
        content = self.findChild(CollapsablePanelContent)
        if content.isVisible():
            content.hide()
        else:
            content.show()

    def set_collapse(self, collapse):
        content = self.findChild(CollapsablePanelContent)
        if collapse:
            content.hide()
        else:
            content.show()


class CollapsablePanelHeader(QLabel):

    def mousePressEvent(self, QMouseEvent):
        super(CollapsablePanelHeader, self).mousePressEvent(QMouseEvent)
        self.parent().toggle()


class CollapsablePanelContent(QFrame):

    def __init__(self, parent=None):
        super(CollapsablePanelContent, self).__init__(parent)


class TextConfigEditor(QPlainTextEdit):

    new_config = pyqtSignal(str)

    def __init__(self, *args):
        super(TextConfigEditor, self).__init__(*args)
        self._timer = TimedUpdate(self._timer_event)
        self.textChanged.connect(lambda: self._timer.add_delayed_callback(400))

    @pyqtSlot()
    def _timer_event(self):
        self.new_config.emit(self.toPlainText())


class TimedUpdate(QTimer):

    def __init__(self, update_cb):
        """Creates a timer that can delay running a given callback function.

        Every time the user adds a delayed callback the timer gets reset to the new value and we will
        wait that new value until calling the callback with the last data given.

        Args:
            update_cb (function): the function we would like to run after a timer has run out
        """
        super(TimedUpdate, self).__init__()
        self._cb_values = []
        self._update_cb = update_cb
        self.timeout.connect(self._call_update_cb)
        self.timeout.connect(self.stop)

    def add_delayed_callback(self, delay, *cb_values):
        """Pushes a new delay to calling the callback function.

        Args:
            delay (int): the time in ms to wait
            cb_values (*list): the list of values to use as arguments to the callback function. Leave empty to disable.
        """
        self._cb_values = cb_values
        self.start(delay)

    def _call_update_cb(self):
        if self._cb_values:
            self._update_cb(*self._cb_values)
        else:
            self._update_cb()


class MapsReorderer(QListWidget):

    items_reordered = pyqtSignal()

    def __init__(self, *args):
        super(MapsReorderer, self).__init__(*args)
        self.installEventFilter(self)

    def eventFilter(self, sender, event):
        if event.type() == QEvent.ChildRemoved:
            self.items_reordered.emit()
        return False


# Regular expression to find floats. Match groups are the whole string, the
# whole coefficient, the decimal part of the coefficient, and the exponent
# part. Copied partly from https://gist.github.com/jdreaver/0be2e44981159d0854f5
_float_re = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')


class FloatValidator(QValidator):

    def __init__(self, *args):
        super(FloatValidator, self).__init__(*args)

    def validate(self, string, position):
        if FloatValidator.valid_float_string(string):
            return QValidator.Acceptable, string, position
        if string == "" or string[position-1] in 'e.-+':
            return QValidator.Intermediate, string, position
        return QValidator.Invalid, string, position

    def fixup(self, text):
        match = _float_re.search(text)
        return match.groups()[0] if match else ""

    @staticmethod
    def valid_float_string(string):
        match = _float_re.search(string)
        return match.groups()[0] == string if match else False


class ScientificDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super(ScientificDoubleSpinBox, self).__init__(*args, **kwargs)
        self.validator = FloatValidator()
        self.setDecimals(1000)

    def validate(self, text, position):
        return self.validator.validate(text, position)

    def fixup(self, text):
        return self.validator.fixup(text)

    def valueFromText(self, text):
        return float(text)

    def textFromValue(self, value):
        def format_float(value):
            """Modified form of the 'g' format specifier."""
            string = "{:g}".format(value).replace("e+", "e")
            string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
            return string

        return format_float(value)

    def stepBy(self, steps):
        text = self.cleanText()
        groups = _float_re.search(text).groups()
        decimal = float(groups[1])
        decimal += steps
        new_string = "{:g}".format(decimal) + (groups[3] if groups[3] else "")
        self.lineEdit().setText(new_string)

