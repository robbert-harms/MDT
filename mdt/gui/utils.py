import time
from functools import wraps
from PyQt5.QtCore import QObject, pyqtSignal
from mdt.log_handlers import LogListenerInterface

__author__ = 'Robbert Harms'
__date__ = "2015-08-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def center_window(q_app, window):
    """Center the given window on the screen.

    Args:
        q_app (QApplication): for desktop information
        window (QMainWindow): the window to center
    """
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
