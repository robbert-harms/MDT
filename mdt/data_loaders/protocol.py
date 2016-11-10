import os
import six
from mdt.protocols import Protocol, load_protocol, auto_load_protocol

__author__ = 'Robbert Harms'
__date__ = "2015-08-25"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def autodetect_protocol_loader(data_source):
    """A function to get a protocol loader using the given data source.

    This tries to do auto detecting for the following data sources:

        - :class:`ProtocolLoader`
        - strings (filename or directory)
        - functions
        - protocol objects

    If a directory is given we try to auto use the protocol from sources in that directory.

    Args:
        data_source: the data source from which to get a protocol loader

    Returns:
        ProtocolLoader: a protocol loader instance.
    """
    if isinstance(data_source, ProtocolLoader):
        return data_source
    elif isinstance(data_source, six.string_types):
        if os.path.isfile(data_source):
            return ProtocolFromFileLoader(data_source)
        else:
            return ProtocolFromDirLoader(data_source)
    elif hasattr(data_source, '__call__'):
        return ProtocolFromFunctionLoader(data_source)
    elif isinstance(data_source, Protocol):
        return ProtocolDirectLoader(data_source)
    raise ValueError('The given data source could not be recognized.')


class ProtocolLoader(object):
    """Interface for loading protocols from different sources."""

    def get_protocol(self):
        """The public method used to get an instance of a protocol.

        Returns:
            Protocol: a protocol object
        """


class ProtocolFromFileLoader(ProtocolLoader):

    def __init__(self, filename):
        """Loads a protocol from the given filename.

        Args:
            filename (str): the filename to use the protocol from.
        """
        super(ProtocolFromFileLoader, self).__init__()
        self._filename = filename
        self._protocol = None

    def get_protocol(self):
        if self._protocol is None:
            self._protocol = load_protocol(self._filename)
        return self._protocol


class ProtocolFromDirLoader(ProtocolLoader):

    def __init__(self, directory):
        """Loads a protocol from the given filename.

        Args:
            directory (str): the directory to use the protocol from.
        """
        super(ProtocolFromDirLoader, self).__init__()
        self._directory = directory
        self._protocol = None

    def get_protocol(self):
        if self._protocol is None:
            self._protocol = auto_load_protocol(self._directory)
        return self._protocol


class ProtocolDirectLoader(ProtocolLoader):

    def __init__(self, protocol):
        """Adapter for returning an already loaded protocol.

        Args:
            protocol (Protocol): the loaded protocol to return.
        """
        super(ProtocolDirectLoader, self).__init__()
        self._protocol = protocol

    def get_protocol(self):
        return self._protocol


class ProtocolFromFunctionLoader(ProtocolLoader):

    def __init__(self, func):
        """Load a protocol from a callback function.

        This class may apply caching.

        Args:
            func: the callback function to call on the moment the protocol is to be loaded.
        """
        super(ProtocolFromFunctionLoader, self).__init__()
        self._func = func
        self._protocol = None

    def get_protocol(self):
        if self._protocol is None:
            self._protocol = self._func()
        return self._protocol



