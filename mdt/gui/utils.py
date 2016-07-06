from collections import OrderedDict
from functools import wraps
import mot.cl_environments
from mdt.configuration import config as mdt_config
from mdt.log_handlers import LogListenerInterface
from mdt.utils import MetaOptimizerBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-08-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_cl_environments_ordered_dict():
    """Get an ordered dictionary with all the CL environments.

    Returns:
        OrderedDict: an ordered dict with all the CL environments
    """
    cl_environments = mot.cl_environments.CLEnvironmentFactory.all_devices()
    cl_env_dict = OrderedDict()
    for ind, env in enumerate(cl_environments):
        s = repr(ind) + ') ' + str(env)
        cl_env_dict.update({s: env})
    return cl_env_dict


class OptimOptions(object):

    optim_routines = {'Powell\'s method': 'Powell',
                      'Nelder-Mead Simplex': 'NMSimplex',
                      'Levenberg Marquardt': 'LevenbergMarquardt',}

    cl_environments = get_cl_environments_ordered_dict()

    def __init__(self):
        self.use_model_default_optimizer = True
        self.double_precision = False

        self.optimizer = mdt_config['optimization_settings']['general']['optimizers'][0]['name']
        self.patience = mdt_config['optimization_settings']['general']['optimizers'][0]['patience']

        self.recalculate_all = False
        self.extra_optim_runs = mdt_config['optimization_settings']['general']['extra_optim_runs']

        self.cl_envs_indices = self._get_prefered_device_indices()
        self.noise_std = 'auto'

    def _get_prefered_device_indices(self):
        l = [ind for ind, env in enumerate(self.cl_environments.values()) if env.is_gpu]
        if l:
            return l
        return [0]

    def get_meta_optimizer_config(self):
        return {
            'optimizers': [{'name': self.optimizer, 'patience': self.patience}],
            'extra_optim_runs': self.extra_optim_runs,
            'extra_optim_runs_apply_smoothing': False,
            'load_balancer': {'name': 'EvenDistribution'},
            'cl_devices': self.cl_envs_indices,
        }

    def get_optimizer(self):
        optimizer_config = self.get_meta_optimizer_config()
        return MetaOptimizerBuilder(optimizer_config).construct()


class ProtocolOptions(object):

    def __init__(self):
        """Simple container class for storing and passing protocol options."""
        self.estimate_sequence_timings = False
        self.seq_timings_units = 'ms'
        self.maxG = 0.04
        self.Delta = None
        self.delta = None
        self.TE = None

        self.extra_column_names = ['Delta', 'delta', 'TE']


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
