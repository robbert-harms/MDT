import os
import time
from contextlib import contextmanager
from functools import wraps

from PyQt5.QtCore import QObject, pyqtSignal, QFileSystemWatcher, pyqtSlot
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from mdt.__version__ import __version__
from mdt.nifti import yield_nifti_info
from mdt.log_handlers import LogListenerInterface

__author__ = 'Robbert Harms'
__date__ = "2015-08-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class QtManager(object):

    windows = []

    @staticmethod
    def get_qt_application_instance():
        q_app = QApplication.instance()
        if q_app is None:
            q_app = QApplication([])
            q_app.lastWindowClosed.connect(QtManager.empty_windows_list)
        return q_app

    @staticmethod
    def exec_():
        if QtManager.windows:
            QtManager.get_qt_application_instance().exec_()

    @staticmethod
    def add_window(window):
        QtManager.windows.append(window)

    @staticmethod
    def empty_windows_list():
        QtManager.windows = []


def center_window(window):
    """Center the given window on the screen.

    Args:
        q_app (QApplication): for desktop information
        window (QMainWindow): the window to center
    """
    q_app = QApplication.instance()

    frame_gm = window.frameGeometry()
    screen = q_app.desktop().screenNumber(q_app.desktop().cursor().pos())
    center_point = q_app.desktop().screenGeometry(screen).center()
    frame_gm.moveCenter(center_point)
    window.move(frame_gm.topLeft())


def function_message_decorator(header, footer):
    """This creates and returns a decorator that prints a header and footer before executing the function.

    Args:
        header (str): the header text, we will add extra decoration to it
        footer (str): the footer text, we will add extra decoration to it

    Returns:
        decorator function
    """
    def _called_decorator(dec_func):

        @wraps(dec_func)
        def _decorator(*args, **kwargs):
            print('')
            print(header)
            print('-'*20)

            response = dec_func(*args, **kwargs)

            print('-'*20)
            print(footer)

            return response
        return _decorator

    return _called_decorator


@contextmanager
def blocked_signals(*widgets):
    """Small context in which the signals of the given widget are blocked.

    Args:
        widgets (QWidget): one or more widgets
    """
    def apply_block(bool_val):
        for w in widgets:
            w.blockSignals(bool_val)

    apply_block(True)
    yield
    apply_block(False)


def print_welcome_message():
    """Prints a small welcome message for after the GUI has loaded.

    This prints to stdout. We expect the GUI to catch the stdout events and redirect them to the GUI.
    """
    from mdt import VERSION
    print('Welcome to MDT version {}.'.format(VERSION))
    print('')
    print('This area is reserved for log output.')
    print('-------------------------------------')


class ForwardingListener(LogListenerInterface):

    def __init__(self, queue):
        """Forwards all incoming messages to the given _logging_update_queue.

        Instances of this class can be used as a log listener to the MDT LogDispatchHandler and as a
        sys.stdout replacement.

        Args:
            queue (Queue): the _logging_update_queue to forward the messages to
        """
        self._queue = queue

    def emit(self, record, formatted_message):
        self._queue.put(formatted_message + "\n")

    def write(self, string):
        self._queue.put(string)


image_files_filters = ['Nifti (*.nii *.nii.gz)',
                       'IMG, HDR (*.img)',
                       'All files (*)']
protocol_files_filters = ['MDT protocol (*.prtcl)',
                          'Text files (*.txt)',
                          'All files (*)']


class UpdateDescriptor(object):

    def __init__(self, attribute_name):
        """Descriptor that will emit a state_updated_signal at each update.

        This accesses from the instance the attribute name prepended with an underscore (_).
        """
        self._attribute_name = attribute_name

    def __get__(self, instance, owner):
        return getattr(instance, '_' + self._attribute_name)

    def __set__(self, instance, value):
        setattr(instance, '_' + self._attribute_name, value)
        instance.state_updated_signal.emit(self._attribute_name)


class MessageReceiver(QObject):

    text_message_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, queue, *args, **kwargs):
        """A QObject (to be run in a QThread) which sits waiting for data to come through a Queue.Queue().

        It blocks until data is available, and one it has got something from the _logging_update_queue, it sends
        it to the "MainThread" by emitting a Qt Signal.

        Attributes:
            is_running (boolean): set to False to stop the receiver.
        """
        super(MessageReceiver, self).__init__(*args, **kwargs)
        self.queue = queue
        self.is_running = True

    def run(self):
        while self.is_running:
            if not self.queue.empty():
                self.text_message_signal.emit(self.queue.get())
            time.sleep(0.001)
        self.finished.emit()


class MainTab(object):

    def tab_opened(self):
        """Called when this tab is selected by the user."""


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


def get_script_file_header_text(info_fields=None):
    """Get the header file text to use when outputting script files matching GUI actions.

    Args:
        info_fields (dict): fields holding information. Common fields are 'Purpose' and 'Example usage'.
            These fields are appended to the header text.

    Returns:
        str: the header (comment) text for the automatically generated script.
    """
    header_text = '# This script was automatically generated by the MDT GUI. \n'
    header_text += '# \n'
    header_text += '# {}: {}\n'.format('Generated on', time.strftime("%Y-%m-%d %H:%M:%S"))
    header_text += '# {}: {}\n'.format('MDT version', __version__)

    for key, value in info_fields.items():
        header_text += '# {key}: {value} \n'.format(key=key, value=value)

    return header_text[:-1]


def split_long_path_elements(original_path, max_single_element_length=25):
    """Split long path elements into smaller ones using spaces

    Args:
        original_path (str): the path you want to split
        max_single_element_length (int): the maximum length allowed per path component (folders and filename).

    Returns:
        str: the same path but with spaces in long path elements. The result will no longer be a valid path.
    """

    def split(p):
        listing = []

        def _split(el):
            if el:
                head, tail = os.path.split(el)
                if not tail:
                    listing.append(head)
                else:
                    _split(head)
                    listing.append(tail)

        _split(p)
        return listing

    elements = list(split(original_path))
    new_elements = []

    for el in elements:
        if len(el) > max_single_element_length:
            item = ''
            for i in range(0, len(el), max_single_element_length):
                item += el[i:i + max_single_element_length] + ' '
            item = item[:-1]
            new_elements.append(item)
        else:
            new_elements.append(el)

    return os.path.join(*new_elements)
