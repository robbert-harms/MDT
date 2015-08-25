import six
from mdt.protocols import Protocol, load_protocol

__author__ = 'Robbert Harms'
__date__ = "2015-08-25"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"



def autodetect_protocol_loader(data_source):
    """A function to get a protocol loader using the given data source.

    This tries to do auto detecting for the following data sources:
        - ProtocolLoader
        - strings (filenames)
        - functions
        - protocol objects

    Args:
        data_source: the data source from which to get a protocol loader

    Returns:
        ProtocolLoader: a protocol loader instance.
    """
    if isinstance(data_source, ProtocolLoader):
        return data_source
    elif isinstance(data_source, six.string_types):
        return ProtocolFromFileLoader(data_source)
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

        This class may apply caching.

        Args:
            filename (str): the filename to load the protocol from.
        """
        self._filename = filename
        self._protocol = None

    def get_protocol(self):
        if self._protocol is None:
            self._protocol = load_protocol(self._filename)
        return self._protocol


class ProtocolDirectLoader(ProtocolLoader):

    def __init__(self, protocol):
        """Adapter for returning an already loaded protocol.

        Args:
            protocol (Protocol): the loaded protocol to return.
        """
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
        self._func = func
        self._protocol = None

    def get_protocol(self):
        if self._protocol is None:
            self._protocol = self._func()
        return self._protocol



