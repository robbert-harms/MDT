from collections import OrderedDict
from functools import wraps
import mdt
from mdt.configuration import config as mdt_config
from mdt.utils import check_user_components, MetaOptimizerBuilder
import mot.cl_environments

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

    smoothing_routines = {'Median filter': 'MedianFilter',
                          'Mean filter': 'MeanFilter'}

    cl_environments = get_cl_environments_ordered_dict()

    def __init__(self):
        self.optimizer = mdt_config['optimization_settings']['general']['optimizers'][0]['name']
        self.patience = mdt_config['optimization_settings']['general']['optimizers'][0]['patience']

        self.smoother = mdt_config['optimization_settings']['general']['smoothing_routines'][0]['name']
        self.smoother_size = mdt_config['optimization_settings']['general']['smoothing_routines'][0]['size']

        self.recalculate_all = True
        self.extra_optim_runs = mdt_config['optimization_settings']['general']['extra_optim_runs']

        self.cl_envs_indices = self._get_prefered_device_indices()

    def _get_prefered_device_indices(self):
        l = [ind for ind, env in enumerate(self.cl_environments.values()) if env.is_gpu]
        if l:
            return l
        return [0]

    def get_optimizer(self):
        optimizer_config = {
            'optimizers': [{'name': self.optimizer, 'patience': self.patience}],
            'extra_optim_runs': self.extra_optim_runs,
            'extra_optim_runs_apply_smoothing': True,
            'smoothing_routines': [{'name': self.smoother, 'size': self.smoother_size}],
            'load_balancer': {'name': 'EvenDistribution'},
            'cl_devices': self.cl_envs_indices,
        }
        return MetaOptimizerBuilder(optimizer_config).construct()


class ProtocolOptions(object):

    def __init__(self):
        """Simple container class for storing and passing protocol options."""
        self.estimate_sequence_timings = False
        self.maxG = 0.04
        self.Delta = None
        self.delta = None
        self.TE = None

        self.extra_column_names = ['Delta', 'delta', 'TE']


def print_welcome_message():
    """Prints a small welcome message for after the GUI has loaded.

    This prints to stdout. We expect the GUI to catch the stdout events and redirect them to the GUI.
    """
    from mdt import VERSION
    welcome_str = 'Welcome to MDT version ' + VERSION + '.'
    print(welcome_str)
    print('')
    print('This area is reserved for print and log output.')
    print('-----------------------------------------------')


def update_user_settings():
    """Updates the user settings if necessary.

    This prints a message to stdout if it updates the users settings.
    We expect the GUI to catch the stdout events and redirect them to the GUI.
    """
    if not check_user_components():
        print('')
        print('Your configuration folder is not up to date. We will create a backup\nof your old settings and '
              'initialize your config directory to the latest version.')
        print('')
        print('...')
        mdt.initialize_user_settings()
        print('')
        print('Initializing home directory completed.')


def function_message_decorator(header, footer):
    """This creates and returns a decorator that prints a header and footer before executing the function.

    Args:
        header (str): the header text, we will add extra decoration to it
        foot (str): the footer text, we will add extra decoration to it

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