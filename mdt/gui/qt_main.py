import sys

import signal

from mdt.gui.qt.tabs.generate_brain_mask_tab import GenerateBrainMaskTab
from mdt.gui.qt.tabs.generate_protocol_tab import GenerateProtocolTab
from mdt.gui.qt.tabs.generate_roi_mask_tab import GenerateROIMaskTab
from mdt.gui.qt.tabs.view_results_tab import ViewResultsTab

try:
    #python 2.7
    from Queue import Queue
except ImportError:
    # python 3.4
    from queue import Queue
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, QTimer, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QStyleFactory
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
        self._computations_thread = ComputationsThread(self)
        self._computations_thread.start()

        self._stdout_old = sys.stdout
        self._stderr_old = sys.stderr
        self._logging_update_queue = Queue()
        self._logging_update_thread = QThread()

        self._message_receiver = MessageReceiver(self._logging_update_queue)
        self._message_receiver.text_message_signal.connect(self.update_log)

        self._message_receiver.moveToThread(self._logging_update_thread)
        self._logging_update_thread.started.connect(self._message_receiver.run)
        self._logging_update_thread.start()
        self._connect_output_textbox()

        self.actionExit.setShortcuts(['Ctrl+q', 'Ctrl+w'])
        self.action_saveLog.triggered.connect(self.save_log)
        self._center()

        self.executionStatusLabel.setText('Idle')
        self.executionStatusIcon.setPixmap(QtGui.QPixmap(":/gui_single/icon_status_red.png"))

        self.generate_mask_tab = GenerateBrainMaskTab(shared_state, self._computations_thread)
        self.generate_mask_tab.setupUi(self.generateBrainMaskTab)

        self.view_results_tab = ViewResultsTab(shared_state, self._computations_thread)
        self.view_results_tab.setupUi(self.viewResultsTab)

        self.generate_roi_mask_tab = GenerateROIMaskTab(shared_state, self._computations_thread)
        self.generate_roi_mask_tab.setupUi(self.generateROIMaskTab)

        self.generate_protocol_tab = GenerateProtocolTab(shared_state, self._computations_thread)
        self.generate_protocol_tab.setupUi(self.generateProtocolTab)

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

    def send_sigint(self, *args):
        self.close()

    @pyqtSlot()
    def save_log(self):
        name = QFileDialog.getSaveFileName(self, 'Save log', directory=self._shared_state.base_dir,
                                           filter='Text file (*.txt);;All files (*)')[0]
        if name:
            with open(name, 'w') as file:
                text = self.loggingTextBox.toPlainText()
                file.write(text)

    @pyqtSlot()
    def computations_started(self):
        self.executionStatusLabel.setText('Computing')
        self.executionStatusIcon.setPixmap(QtGui.QPixmap(":/gui_single/icon_status_green.png"))

    @pyqtSlot()
    def computations_finished(self):
        self.executionStatusLabel.setText('Idle')
        self.executionStatusIcon.setPixmap(QtGui.QPixmap(":/gui_single/icon_status_red.png"))

    @pyqtSlot(str)
    def update_log(self, string):
        sb = self.loggingTextBox.verticalScrollBar()
        scrollbar_position = sb.value()
        autoscroll = scrollbar_position == sb.maximum()
        self.loggingTextBox.moveCursor(QtGui.QTextCursor.End)
        self.loggingTextBox.insertPlainText(string)

        if autoscroll:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(scrollbar_position)


class ComputationsThread(QThread):

    def __init__(self, main_window, *args, **kwargs):
        """This is the thread handler for the computations.

        When running computations using this thread please connect signals to the starting and finished slot of this
        class. These handlers notify the main window of the computations.
        """
        super(ComputationsThread, self).__init__(*args, **kwargs)
        self.main_window = main_window

    @pyqtSlot()
    def starting(self):
        self.main_window.computations_started()

    @pyqtSlot()
    def finished(self):
        self.main_window.computations_finished()


def start_single_model_gui(base_dir=None, action=None):
    """Start the single model GUI.

    Args:
        base_dir (str): the starting directory for all file opening actions
        action (str): an action command for opening tabs and files. Possible actions:
            - view_maps: opens the view maps tab and opens the base_dir

    """
    state = SharedState()
    state.base_dir = base_dir

    app = QApplication([])

    # catches the sigint
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    single_model_gui = MDTGUISingleModel(app, state)
    signal.signal(signal.SIGINT, single_model_gui.send_sigint)

    single_model_gui.show()

    if action == 'view_maps':
        single_model_gui.MainTabs.setCurrentIndex(4)
        single_model_gui.view_results_tab.open_dir(base_dir)

    sys.exit(app.exec_())


if __name__ == '__main__':
    start_single_model_gui()
