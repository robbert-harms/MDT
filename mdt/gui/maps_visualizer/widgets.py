from PyQt5.QtCore import QEvent
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
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
        self.textChanged.connect(self._timed_update)
        self._timer = QTimer()
        self._timer.timeout.connect(self._timer_event)
        self._timer.timeout.connect(self._timer.stop)

    @pyqtSlot()
    def _timed_update(self):
        self._timer.start(400)

    @pyqtSlot()
    def _timer_event(self):
        self.new_config.emit(self.toPlainText())


class MapsReorderer(QListWidget):

    items_reordered = pyqtSignal()

    def __init__(self, *args):
        super(MapsReorderer, self).__init__(*args)
        self.installEventFilter(self)

    def eventFilter(self, sender, event):
        if event.type() == QEvent.ChildRemoved:
            self.items_reordered.emit()
        return False
