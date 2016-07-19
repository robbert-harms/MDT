from functools import wraps
from mdt.log_handlers import LogListenerInterface

__author__ = 'Robbert Harms'
__date__ = "2015-08-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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
