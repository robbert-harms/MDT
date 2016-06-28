import time
from PyQt5.QtCore import QObject, pyqtSignal

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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


class SharedState(QObject):

    state_updated_signal = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        """The shared state for the single model GUI

        Attributes:
            base_dir (str): the base dir for all file opening operations
            dimension_index (int): the dimension index used in various operations
            slice_index (int): the slice index used in various operations
        """
        super(SharedState, self).__init__(*args, **kwargs)

        shared_attributes = {'base_dir': None,
                             'dimension_index': 0,
                             'slice_index': 0}

        for key, value in shared_attributes.items():
            setattr(self, '_' + key, value)
            setattr(SharedState, key, UpdateDescriptor(key))
            setattr(self, 'set_' + key, lambda v: setattr(self, key, v))


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
