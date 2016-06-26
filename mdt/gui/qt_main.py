import sys
try:
    #python 2.7
    from Queue import Queue
except ImportError:
    # python 3.4
    from queue import Queue
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMainWindow, QApplication
from mdt.gui.qt.design.ui_gui_single import Ui_MainWindow
from mdt.gui.qt.tabs import ViewResultsTab
from mdt.gui.qt.utils import MessageReceiver
from mdt.gui.utils import print_welcome_message, ForwardingListener
from mdt.log_handlers import LogDispatchHandler

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MDTGUISingleModel(QMainWindow, Ui_MainWindow):

    def __init__(self, q_app, initial_directory=None):
        super(MDTGUISingleModel, self).__init__()
        self.setupUi(self)
        self.q_app = q_app
        self.initial_directory = initial_directory

        self._stdout_old = sys.stdout
        self._stderr_old = sys.stderr
        self.queue = Queue()

        thread = QThread(self)

        self.message_receiver = MessageReceiver(self.queue)
        self.message_receiver.text_message_signal.connect(self.loggingTextBox.insertPlainText)

        self.message_receiver.moveToThread(thread)
        thread.started.connect(self.message_receiver.run)
        thread.start()
        self._connect_output_textbox()

        self.actionExit.setShortcuts(['Ctrl+q', 'Ctrl+w'])
        self._center()

        ViewResultsTab(initial_directory).setupUi(self.viewResultsTab)

    def _connect_output_textbox(self):
        sys.stdout = ForwardingListener(self.queue)
        sys.stderr = ForwardingListener(self.queue)
        LogDispatchHandler.add_listener(ForwardingListener(self.queue))
        print_welcome_message()

    def closeEvent(self, event):
        sys.stdout = self._stdout_old
        sys.stderr = self._stderr_old
        super(MDTGUISingleModel, self).closeEvent(event)

    def _center(self):
        frameGm = self.frameGeometry()
        screen = self.q_app.desktop().screenNumber(self.q_app.desktop().cursor().pos())
        centerPoint = self.q_app.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())


def start_single_model_gui(initial_directory=None):
    app = QApplication([])
    single_model_gui = MDTGUISingleModel(app, initial_directory)
    single_model_gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    start_single_model_gui()
