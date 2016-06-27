import sys

import signal

from mdt.gui.qt.tabs.generate_brain_mask_tab import GenerateBrainMaskTab
from mdt.gui.qt.tabs.view_results_tab import ViewResultsTab

try:
    #python 2.7
    from Queue import Queue
except ImportError:
    # python 3.4
    from queue import Queue
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication
from mdt.gui.qt.design.ui_gui_single import Ui_MainWindow
from mdt.gui.qt.utils import MessageReceiver, SharedState
from mdt.gui.utils import print_welcome_message, ForwardingListener
from mdt.log_handlers import LogDispatchHandler

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MDTGUISingleModel(QMainWindow, Ui_MainWindow):

    def __init__(self, q_app, shared_state):
        super(MDTGUISingleModel, self).__init__()
        self.setupUi(self)
        self._q_app = q_app
        self._shared_state = shared_state
        self._computations_thread = QThread(self)

        self._stdout_old = sys.stdout
        self._stderr_old = sys.stderr
        self._logging_update_queue = Queue()
        self._logging_update_thread = QThread(self)

        self._message_receiver = MessageReceiver(self._logging_update_queue)
        self._message_receiver.text_message_signal.connect(self.loggingTextBox.insertPlainText)

        self._message_receiver.moveToThread(self._logging_update_thread)
        self._logging_update_thread.started.connect(self._message_receiver.run)
        self._logging_update_thread.start()
        self._connect_output_textbox()

        self.actionExit.setShortcuts(['Ctrl+q', 'Ctrl+w'])
        self._center()

        GenerateBrainMaskTab(shared_state, self._computations_thread).setupUi(self.generateBrainMaskTab)
        ViewResultsTab(shared_state, self._computations_thread).setupUi(self.viewResultsTab)

    def _connect_output_textbox(self):
        sys.stdout = ForwardingListener(self._logging_update_queue)
        sys.stderr = ForwardingListener(self._logging_update_queue)
        LogDispatchHandler.add_listener(ForwardingListener(self._logging_update_queue))
        print_welcome_message()

    def closeEvent(self, event):
        sys.stdout = self._stdout_old
        sys.stderr = self._stderr_old
        self._message_receiver.is_running = False
        self._logging_update_thread.quit()
        self._logging_update_thread.wait(10)
        super(MDTGUISingleModel, self).closeEvent(event)

    def _center(self):
        frameGm = self.frameGeometry()
        screen = self._q_app.desktop().screenNumber(self._q_app.desktop().cursor().pos())
        centerPoint = self._q_app.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())


def sigint_handler(*args):
    

def start_single_model_gui(base_dir=None):
    signal.signal(signal.SIGINT, sigint_handler)
    state = SharedState()
    state.base_dir = base_dir

    app = QApplication([])

    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    single_model_gui = MDTGUISingleModel(app, state)
    single_model_gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    start_single_model_gui()
