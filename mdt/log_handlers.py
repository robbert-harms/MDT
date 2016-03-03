import codecs
import logging
import os
import sys

__author__ = 'Robbert Harms'
__date__ = "2015-08-19"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelOutputLogHandler(logging.StreamHandler):

    __instances__ = set()

    def __init__(self, mode='a', encoding=None):
        """This logger logs information about a model optimization to the folder of the model that is being optimized.
        """
        super(ModelOutputLogHandler, self).__init__()
        self.__class__.__instances__.add(self)

        if codecs is None:
            encoding = None

        self._output_file = None
        self.mode = mode
        self.encoding = encoding
        self.stream = None

    @property
    def output_file(self):
        return self._output_file

    @output_file.setter
    def output_file(self, output_file):
        self.close()

        self._output_file = output_file
        if self._output_file:
            if not os.path.isdir(os.path.dirname(self._output_file)):
                os.makedirs(os.path.dirname(self._output_file))
            self._open()

    def emit(self, record):
        if self._output_file and self.stream:
            super(ModelOutputLogHandler, self).emit(record)

    def close(self):
        if self._output_file:
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
        if self._output_file:
            if self.encoding is None:
                self.stream = open(self._output_file, self.mode)
            else:
                self.stream = codecs.open(self._output_file, self.mode, self.encoding)


class StdOutHandler(logging.StreamHandler):

    def __init__(self, stream=None):
        """A redirect for stdout.

        This catches the errors if stdout is None.
        """
        stream = stream or sys.stdout
        super(StdOutHandler, self).__init__(stream=stream)

    def emit(self, record):
        if self.stream:
            super(StdOutHandler, self).emit(record)


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
        """Add a listener to this handler.

        Args:
            listener (LogListenerInterface): listener that implements the log listener interface.

        Returns:
            int: the listener id number. You can use this to remove the listener again.
        """
        listener_id = len(LogDispatchHandler._listeners)
        LogDispatchHandler._listeners.append(listener)
        return listener_id

    @staticmethod
    def remove_listener(listener_id):
        del LogDispatchHandler._listeners[listener_id]


class LogListenerInterface(object):
    """Interface for listeners to work in conjunction with LogDispatchHandler"""

    def emit(self, record, formatted_message):
        pass
