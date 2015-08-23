import codecs
import logging
import os
import tempfile

__author__ = 'Robbert Harms'
__date__ = "2015-08-19"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelOutputLogHandler(logging.StreamHandler):

    output_file = tempfile.mkstemp()[1]

    def __init__(self, mode='a', encoding=None):
        """This logger can log information about a model optimization to the folder of the model being optimized.

        One can change the class attribute 'output_file' to change the file items are logged to.
        """
        super(ModelOutputLogHandler, self).__init__()

        if codecs is None:
            encoding = None
        self.baseFilename = os.path.abspath(ModelOutputLogHandler.output_file)
        self.mode = mode
        self.encoding = encoding
        self.stream = None
        self._open()

    def emit(self, record):
        if ModelOutputLogHandler.output_file is not None:
            if os.path.abspath(ModelOutputLogHandler.output_file) != self.baseFilename:
                self.close()
                self.baseFilename = os.path.abspath(ModelOutputLogHandler.output_file)

            if self.stream is None:
                self.stream = self._open()

            super(ModelOutputLogHandler, self).emit(record)

    def close(self):
        """Closes the stream."""
        self.acquire()
        try:
            if self.stream:
                self.flush()
                if hasattr(self.stream, "close"):
                    self.stream.close()
                self.stream = None
            super(ModelOutputLogHandler, self).close()
        finally:
            self.release()

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        if self.encoding is None:
            stream = open(self.baseFilename, self.mode)
        else:
            stream = codecs.open(self.baseFilename, self.mode, self.encoding)
        return stream

    @staticmethod
    def reset_output_file():
        ModelOutputLogHandler.output_file = tempfile.mkstemp()[1]


class LogDispatchHandler(logging.StreamHandler):
    """This class is able to dispatch messages to all the attached log listeners.

    You can add listeners by adding them to the list of listeners. This list is a class variable and as such is
    available to all instances and subclasses.

    The listeners should be of instance LogListenerInterface.
    """
    _listeners = []

    def emit(self, record):
        for listener in self._listeners:
            listener.emit(record, self.format(record))

    @staticmethod
    def add_listener(listener):
        LogDispatchHandler._listeners.append(listener)


class LogListenerInterface(object):

    def emit(self, record, formatted_message):
        pass
