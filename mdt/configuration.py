import os
import re
from copy import deepcopy

import collections
import yaml
from contextlib import contextmanager
from pkg_resources import resource_stream
from six import string_types

from mot.factory import get_optimizer_by_name
from mdt.components_loader import ProcessingStrategiesLoader, NoiseSTDCalculatorsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

"""The config dictionary."""
_config = {}


def load_builtin():
    """Load the config file from the skeleton in mdt/data/mdt.conf"""
    with resource_stream('mdt', 'data/mdt.conf') as f:
        return load_from_yaml(f.read())


def load_user_home():
    """Load the config file from user home directory"""
    from mdt import get_config_dir
    config_file = os.path.join(get_config_dir(), 'mdt.conf')
    if os.path.isfile(config_file):
        with open(config_file) as f:
            return load_from_yaml(f.read())
    raise ValueError('Config file could not be loaded.')


def load_specific(file_name):
    """Can be called by the application to load the config from a specific file.

    This assumes that the given file contains YAML content, that is, we want to process it
    with the function load_from_yaml().

    Please note that the last configuration loaded overwrites the values of the previously loaded config files.

    Args:
        file_name (str): The name of the file to load.
    """
    with open(file_name) as f:
        return load_from_yaml(f.read())


def load_from_yaml(yaml_str):
    """Can be called to load configuration options from a YAML string.

    Please note that the last configuration loaded is the one used.

    Args:
        yaml_str (str): The string containing the YAML config to parse.
    """
    d = yaml.load(yaml_str)
    if d is not None and isinstance(d, dict):
        return d
    raise ValueError('No config dict found in YAML string.')


_config = load_builtin()
try:
    _config = load_user_home()
except ValueError:
    pass


def gzip_optimization_results():
    """Check if we should write the volume maps from the optimization gzipped or not.

    Returns:
        boolean: True if the results of optimization computations should be gzipped, False otherwise.
    """
    return _config['output_format']['optimization']['gzip']


def gzip_sampling_results():
    """Check if we should write the volume maps from the sampling gzipped or not.

    Returns:
        boolean: True if the results of sampling computations should be gzipped, False otherwise.
    """
    return _config['output_format']['optimization']['gzip']


def get_tmp_results_dir():
    return _config['tmp_results_dir']


def get_logging_configuration_dict():
    """Get the configutation dictionary for the logging.dictConfig().

    MDT uses a few special logging configuration options to log to the files and GUI's. These options are defined
    using a configuration dictionary that this function returns.

    Returns:
        dict: the configuration dict for use with dictConfig of the Python logging modules
    """
    return _config['logging']['info_dict']


class OptimizationSettings(object):

    @staticmethod
    def get_optimizer_configs(model_names=None):
        """Get the settings per optimizers.

        Args:
            model_names (list of str): if set we try to match the model specific optimization
                settings to this model name or cascaded list of model names

        Returns:
            list of OptimizerConfig: the optimization configuration objects per defined optimizer for this model.
        """
        settings = [OptimizerConfig(m['name'], m.get('patience', None), m.get('optimizer_options', None)) for m in
                    _config['optimization_settings']['general']['optimizers']]

        if model_names is not None:
            model_config = get_model_config(model_names, _config['optimization_settings']['model_specific'])
            if model_config:
                return [OptimizerConfig(m['name'], m['patience']) for m in model_config['optimizers']]

        return settings

    @staticmethod
    def get_optimizer_names(model_name=None):
        """Get the names of all the optimizers we should use.

        Args:
            model_name (str or list of str): if set we try to match the model specific optimization
                settings to this model name or cascaded list of model names

        Returns:
            list of str: the names of the optimizers we will use during optimization
        """
        return [el.name for el in OptimizationSettings.get_optimizer_configs(model_name)]

    @staticmethod
    def get_extra_optim_runs():
        return int(_config['optimization_settings']['general']['extra_optim_runs'])


class OptimizerConfig(object):

    def __init__(self, name, patience=None, optimizer_options=None):
        """Container object for an optimization routine settings"""
        self.name = name
        self.patience = patience
        self.optimizer_options = optimizer_options

    def build_optimizer(self):
        optimizer = get_optimizer_by_name(self.name)
        return optimizer(patience=self.patience, optimizer_options=self.optimizer_options)


def get_model_config(model_names, config):
    """Get from the given dictionary the config for the given model.

    This tries to find the best match between the given config items (by key) and the given model list. For example
    if model_names is ['BallStick', 'S0'] and we have the following config dict:
        {'^S0$': 0,
         '^BallStick$': 1
         ('^BallStick$', '^S0$'): 2,
         ('^BallStickStick$', '^BallStick$', '^S0$'): 3,
         }

    then this function should return 2. because that one is the best match, even though the last option is also a
    viable match. That is, since a subset of the keys in the last option also matches the model names, it is
    considered a match as well. Still the best match is the third option (returning 2).

    Args:
        model_names (list of str): the names of the models we want to match. This should contain the entire
            recursive list of cascades leading to the single model we want to get the config for.
        config (dict): the config items with as keys either a single model regex for a name or a list of regex for
            a chain of model names.

    Returns:
        The config content of the best matching key.
    """
    if not config:
        return {}

    def get_key_length(key):
        if isinstance(key, tuple):
            return len(key)
        return 1

    def is_match(model_names, config_key):
        if isinstance(model_names, string_types):
            model_names = [model_names]

        if len(model_names) != get_key_length(config_key):
            return False

        if isinstance(config_key, tuple):
            return all([re.match(config_key[ind], model_names[ind]) for ind in range(len(config_key))])

        return re.match(config_key, model_names[0])

    key_length_lookup = ((get_key_length(key), key) for key in config.keys())
    ascending_keys = tuple(item[1] for item in sorted(key_length_lookup, key=lambda info: info[0]))

    # direct matching
    for key in ascending_keys:
        if is_match(model_names, key):
            return config[key]

    # partial matching string keys to last model name
    for key in ascending_keys:
        if not isinstance(key, tuple):
            if is_match(model_names[-1], key):
                return config[key]

    # partial matching tuple keys with a moving filter
    for key in ascending_keys:
        if isinstance(key, tuple):
            for start_ind in range(len(key)):
                sub_key = key[start_ind:]

                if is_match(model_names, sub_key):
                    return config[key]

    # no match found
    return {}


def get_processing_strategy(processing_type, model_names=None):
    """Get the correct processing strategy for the given model.

    Args:
        processing_type (str): 'optimization', 'sampling' or any other of the
            processing_strategies defined in the config
        model_names (list of str): the list of model names (the full recursive cascade of model names)

    Returns:
        ModelProcessingStrategy: the processing strategy to use for this model
    """
    strategy_name = _config['processing_strategies'][processing_type]['general']['name']
    options = _config['processing_strategies'][processing_type]['general'].get('options', {}) or {}

    if model_names and 'model_specific' in _config['processing_strategies'][processing_type]:
        info_dict = get_model_config(model_names, _config['processing_strategies'][processing_type]['model_specific'])

        if info_dict:
            strategy_name = info_dict['name']
            options = info_dict.get('options', {}) or {}

    return ProcessingStrategiesLoader().load(strategy_name, **options)


def get_noise_std_estimators():
    """Get the noise std estimators for finding the std of the noise.

    Returns:
        list of ComplexNoiseStdEstimator: the noise estimators to use for finding the complex noise
    """
    loader = NoiseSTDCalculatorsLoader()
    return [loader.load(c) for c in _config['noise_std_estimating']['general']['estimators']]


@contextmanager
def config_context(config_action):
    """Creates a context in which the config action is applied and unapplies the configuration after execution.

    Args:
        config_action (ConfigAction): the configuration action to use
    """
    config_action.apply()
    yield
    config_action.unapply()


class ConfigAction(object):

    def __init__(self):
        """Defines a configuration action for the use in a configuration context.

        This should define an apply and an unapply function that sets and unsets the given configuration options.

        The applying action needs to remember the state before applying the action.
        """

    def apply(self):
        """Apply the current action to the current runtime configuration."""

    def unapply(self):
        """Reset the current configuration to the previous state."""


class VoidConfigAction(ConfigAction):
    """Does nothing. Meant as a container to not have to check for None's everywhere."""

    def apply(self):
        pass

    def unapply(self):
        pass


class SimpleConfigAction(ConfigAction):

    def __init__(self):
        """Defines a default implementation of a configuration action.

        This simple config implements a default apply() method that saves the current state and a default
        unapply() that restores the previous state.

        It is easiest to implement _apply() for extra actions.
        """
        super(SimpleConfigAction, self).__init__()
        self._old_config = {}

    def apply(self):
        """Apply the current action to the current runtime configuration."""
        self._old_config = deepcopy(_config)
        self._apply()

    def unapply(self):
        """Reset the current configuration to the previous state."""
        global _config
        _config = self._old_config

    def _apply(self):
        """Implement this function add apply() logic after this class saves the current config."""


class YamlStringAction(SimpleConfigAction):

    def __init__(self, yaml_str):
        super(YamlStringAction, self).__init__()
        self._yaml_str = yaml_str

    def _apply(self):
        global _config
        _config = recursive_merge_dict(_config, self._get_dict_from_yaml(), in_place=True)

    def _get_dict_from_yaml(self):
        d = yaml.load(self._yaml_str)
        if d is not None:
            return d
        return {}


def recursive_merge_dict(dictionary, update_dict, in_place=False):
    """ Recursively merge the given dictionary with the new values.

    If in_place is false this does not merge in place but creates new dictionary.

    If update_dict is None we return the original dictionary, or a copy if in_place is False.

    Args:
        dictionary (dict): the dictionary we want to update
        update_dict (dict): the dictionary with the new values
        in_place (boolean): if true, the changes are in place in the first dict.

    Returns:
        dict: a combination of the two dictionaries in which the values of the last dictionary take precedence over
            that of the first.
            Example:
                recursive_merge_dict(
                    {'k1': {'k2': 2}},
                    {'k1': {'k2': {'k3': 3}}, 'k4': 4}
                )

                gives:

                {'k1': {'k2': {'k3': 3}}, 'k4': 4}

            In the case of lists in the dictionary, we do no merging and always use the new value.
    """
    if not in_place:
        dictionary = deepcopy(dictionary)

    if not update_dict:
        return dictionary

    def merge(d, upd):
        for k, v in upd.items():
            if isinstance(v, collections.Mapping):
                r = merge(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = upd[k]
        return d

    return merge(dictionary, update_dict)
