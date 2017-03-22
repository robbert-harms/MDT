"""Contains the runtime configuration of MDT.

This consists of two parts, functions to get the current runtime settings and configuration actions to update these
settings. To set a new configuration, create a new :py:class:`ConfigAction` and use this within a context environment
using :py:func:`config_context`. Example:

.. code-block:: python

    from mdt.configuration import YamlStringAction, config_context

    config = '''
    optimization:
        general:
            optimizers:
                -   name: 'NMSimplex'
                    patience: 10
    '''
    with mdt.config_context(YamlStringAction(config)):
        mdt.fit_model(...)
"""
import os
import re
from copy import deepcopy

import collections
import yaml
from contextlib import contextmanager
from pkg_resources import resource_stream
from six import string_types

from mot.factory import get_optimizer_by_name, get_sampler_by_name, get_proposal_update_by_name
from mdt.components_loader import ProcessingStrategiesLoader, NoiseSTDCalculatorsLoader
import mot.configuration
from mot.load_balance_strategies import EvenDistribution

from mdt.__version__ import __version__

__author__ = 'Robbert Harms'
__date__ = "2015-06-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

""" The current configuration """
_config = {}


def get_config_dir():
    """Get the location of the components.

    Return:
        str: the path to the components
    """
    return os.path.join(os.path.expanduser("~"), '.mdt', __version__)


def config_insert(keys, value):
    """Insert the given value in the given key.

    This will create all layers of the dictionary if needed.

    Args:
        keys (list of str): the position of the input value
        value (object): the value to put at the position of the key.
    """
    config = _config
    for key in keys[:-1]:
        if key not in config:
            config[key] = {}
        config = config[key]

    config[keys[-1]] = value


def ensure_exists(keys):
    """Ensure the given layer of keys exists.

    Args:
        keys (list of str): the positions to ensure exist
    """
    config = _config
    for key in keys:
        if key not in config:
            config[key] = {}
        config = config[key]


def load_builtin():
    """Load the config file from the skeleton in mdt/data/mdt.conf"""
    with resource_stream('mdt', 'data/mdt.conf') as f:
        load_from_yaml(f.read())


def load_user_home():
    """Load the config file from the user home directory"""
    config_file = os.path.join(get_config_dir(), 'mdt.conf')
    if os.path.isfile(config_file):
        with open(config_file) as f:
            load_from_yaml(f.read())
    else:
        raise IOError('Config file could not be loaded.')


def load_user_gui():
    """Load the gui specific config file from the user home directory"""
    config_file = os.path.join(get_config_dir(), 'mdt.gui.conf')
    if os.path.isfile(config_file):
        with open(config_file) as f:
            load_from_yaml(f.read())
    else:
        raise IOError('Config file could not be loaded.')


def load_specific(file_name):
    """Can be called by the application to use the config from a specific file.

    This assumes that the given file contains YAML content, that is, we want to process it
    with the function load_from_yaml().

    Please note that the last configuration loaded overwrites the values of the previously loaded config files.

    Args:
        file_name (str): The name of the file to use.
    """
    with open(file_name) as f:
        load_from_yaml(f.read())


def load_from_yaml(yaml_str):
    """Can be called to use configuration options from a YAML string.

    This will update the current configuration with the new options.

    Args:
        yaml_str (str): The string containing the YAML config to parse.
    """
    config_dict = yaml.safe_load(yaml_str) or {}
    load_from_dict(config_dict)


def load_from_dict(config_dict):
    """Load configuration options from a given dictionary.

    Args:
        config_dict (dict): the dictionary from which to use the configurations
    """
    for key, value in config_dict.items():
        loader = get_section_loader(key)
        loader.load(value)


def update_gui_config(update_dict):
    """Update the GUI configuration file with the given settings.

    Args:
        update_dict (dict): the items to update in the GUI config file
    """
    update_write_config(os.path.join(get_config_dir(), 'mdt.gui.conf'), update_dict)


def update_write_config(config_file, update_dict):
    """Update a given configuration file with updated values.

    If the configuration file does not exist, a new one is created.

    Args:
        config_file (str): the location of the config file to update
        update_dict (dict): the items to update in the config file
    """
    if not os.path.exists(config_file):
        with open(config_file, 'a'):
            pass

    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f.read()) or {}

    for key, value in update_dict.items():
        loader = get_section_loader(key)
        loader.update(config_dict, value)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_dict, f)


class ConfigSectionLoader(object):

    def load(self, value):
        """Load the given configuration value into the current configuration.

        Args:
            value: the value to use in the configuration
        """

    def update(self, config_dict, updates):
        """Update the given configuration dictionary with the values in the given updates dict.

        This enables automating updating a configuration file. Updates are written in place.

        Args:
            config_dict (dict): the current configuration dict
            updates (dict): the updated values to add to the given config dict.
        """


class OutputFormatLoader(ConfigSectionLoader):
    """Loader for the top level key output_format. """

    def load(self, value):
        for item in ['optimization', 'sampling']:
            options = value.get(item, {})

            if 'gzip' in options:
                config_insert(['output_format', item, 'gzip'], bool(options['gzip']))


class LoggingLoader(ConfigSectionLoader):
    """Loader for the top level key logging. """

    def load(self, value):
        ensure_exists(['logging', 'info_dict'])
        if 'info_dict' in value:
            self._load_info_dict(value['info_dict'])

    def _load_info_dict(self, info_dict):
        for item in ['version', 'disable_existing_loggers', 'formatters', 'handlers', 'loggers', 'root']:
            if item in info_dict:
                config_insert(['logging', 'info_dict', item], info_dict[item])


class OptimizationSettingsLoader(ConfigSectionLoader):
    """Loads the optimization section"""

    def load(self, value):
        ensure_exists(['optimization', 'general'])
        ensure_exists(['optimization', 'model_specific'])

        if 'general' in value:
            config_insert(['optimization', 'general'], value['general'])

        if 'model_specific' in value:
            for key, sub_value in value['model_specific'].items():
                config_insert(['optimization', 'model_specific', key], sub_value)


class SampleSettingsLoader(ConfigSectionLoader):
    """Loads the sampling section"""

    def load(self, value):
        ensure_exists(['sampling', 'general'])

        if 'general' in value:
            config_insert(['sampling', 'general'], value['general'])


class ProcessingStrategySectionLoader(ConfigSectionLoader):
    """Loads the config section processing_strategies"""

    def load(self, value):
        if 'optimization' in value:
            self._load_options('optimization', value['optimization'])
        if 'sampling' in value:
            self._load_options('sampling', value['sampling'])

    def _load_options(self, current_type, options):
        if 'general' in options:
            config_insert(['processing_strategies', current_type, 'general'], options['general'])

        ensure_exists(['processing_strategies', current_type, 'model_specific'])
        if 'model_specific' in options:
            for key, value in options['model_specific'].items():
                config_insert(['processing_strategies', current_type, 'model_specific', key], value)


class TmpResultsDirSectionLoader(ConfigSectionLoader):
    """Load the section tmp_results_dir"""

    def load(self, value):
        config_insert(['tmp_results_dir'], value)


class NoiseStdEstimationSectionLoader(ConfigSectionLoader):
    """Load the section noise_std_estimating"""

    def load(self, value):
        if 'estimators' in value:
            config_insert(['noise_std_estimating', 'estimators'], value['estimators'])


class DefaultProposalUpdateLoader(ConfigSectionLoader):
    """Load the default proposal update function."""

    def load(self, value):
        if 'name' in value:
            kwargs = {}
            if 'settings' in value:
                kwargs = value['settings']

            update_class = get_proposal_update_by_name(value['name'])
            mot.configuration.set_default_proposal_update(update_class(**kwargs))


class AutomaticCascadeModels(ConfigSectionLoader):
    """Load the automatic cascade model settings."""

    def load(self, value):
        config_insert(['auto_generate_cascade_models', 'enabled'], value.get('enabled', True))
        config_insert(['auto_generate_cascade_models', 'excluded'], value.get('excluded', []))


class RuntimeSettingsLoader(ConfigSectionLoader):

    def load(self, value):
        if 'cl_device_ind' in value:
            if value['cl_device_ind'] is not None:
                from mdt.utils import get_cl_devices
                all_devices = get_cl_devices()

                indices = value['cl_device_ind']
                if not isinstance(indices, collections.Iterable):
                    indices = [indices]

                devices = [all_devices[ind] for ind in indices if ind < len(all_devices)]

                if devices:
                    mot.configuration.set_cl_environments(devices)
                    mot.configuration.set_load_balancer(EvenDistribution())

    def update(self, config_dict, updates):
        if 'runtime_settings' not in config_dict:
            config_dict.update({'runtime_settings': {}})
        config_dict['runtime_settings'].update(updates)


def get_section_loader(section):
    """Get the section loader to use for the given top level section.

    Args:
        section (str): the section key we want to get the loader for

    Returns:
        ConfigSectionLoader: the config section loader for this top level section of the configuration.
    """
    if section == 'output_format':
        return OutputFormatLoader()

    if section == 'logging':
        return LoggingLoader()

    if section == 'optimization':
        return OptimizationSettingsLoader()

    if section == 'sampling':
        return SampleSettingsLoader()

    if section == 'processing_strategies':
        return ProcessingStrategySectionLoader()

    if section == 'tmp_results_dir':
        return TmpResultsDirSectionLoader()

    if section == 'noise_std_estimating':
        return NoiseStdEstimationSectionLoader()

    if section == 'runtime_settings':
        return RuntimeSettingsLoader()

    if section == 'default_proposal_update':
        return DefaultProposalUpdateLoader()

    if section == 'auto_generate_cascade_models':
        return AutomaticCascadeModels()

    raise ValueError('Could not find a suitable configuration loader for the section {}.'.format(section))


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
    return _config['output_format']['sampling']['gzip']


def get_tmp_results_dir():
    """Get the default tmp results directory.

    This is the default directory for saving temporary computation results. Set to None to disable this and
    use the model directory.

    Returns:
        str or None: the tmp results dir to use during optimization and sampling
    """
    return _config['tmp_results_dir']


def get_processing_strategy(processing_type, model_names=None, **kwargs):
    """Get the correct processing strategy for the given model.

    Args:
        processing_type (str): 'optimization', 'sampling' or any other of the
            processing_strategies defined in the config
        model_names (list of str): the list of model names (the full recursive cascade of model names)
        **kwargs: passed to the constructor of the loaded processing strategy.

    Returns:
        ModelProcessingStrategy: the processing strategy to use for this model
    """
    strategy_name = _config['processing_strategies'][processing_type]['general']['name']
    options = _config['processing_strategies'][processing_type]['general'].get('options', {}) or {}

    if model_names and ('model_specific' in _config['processing_strategies'][processing_type]):
        info_dict = get_model_config(model_names, _config['processing_strategies'][processing_type]['model_specific'])

        if info_dict:
            strategy_name = info_dict['name']
            options = info_dict.get('options', {}) or {}

    options.update(kwargs)

    return ProcessingStrategiesLoader().load(strategy_name, **options)


def get_noise_std_estimators():
    """Get the noise std estimators for finding the std of the noise.

    Returns:
        list of ComplexNoiseStdEstimator: the noise estimators to use for finding the complex noise
    """
    loader = NoiseSTDCalculatorsLoader()
    return [loader.load(c) for c in _config['noise_std_estimating']['estimators']]


def get_logging_configuration_dict():
    """Get the configuration dictionary for the logging.dictConfig().

    MDT uses a few special logging configuration options to log to the files and GUI's. These options are defined
    using a configuration dictionary that this function returns.

    Returns:
        dict: the configuration dict for use with dictConfig of the Python logging modules
    """
    return _config['logging']['info_dict']


def get_general_optimizer():
    """Load the general optimizer from the configuration.

    Returns:
        Optimizer: the configured optimizer for use in MDT
    """
    return _resolve_optimizer(_config['optimization']['general'])


def get_optimizer_for_model(model_names):
    """Get the optimizer for this specific cascade of models.

    This configuration function supports having a different optimizer for optimizing, for example, NODDI in a
    cascade and NODDI without a cascade.

    Args:
        model_names (list of str): the list of model names (typically a cascade of models) for which we
            want to get the optimizer to use.

    Returns:
        Optimizer: the optimizer to use for optimizing the specific model
    """
    info_dict = get_model_config(model_names, _config['optimization']['model_specific'])

    if info_dict:
        return _resolve_optimizer(info_dict)
    else:
        return get_general_optimizer()


def _resolve_optimizer(optimizer_info):
    """Resolve the optimization routine from the given information dictionary.

    Args:
        optimizer_info (dict): the optimization dictionary with at least 'name' for the optimizer and settings
            for the optimizer settings

    Returns:
        optimizer: the optimization routine
    """
    name = optimizer_info['name']
    settings = deepcopy(optimizer_info.get('settings', {}) or {})
    optimizer = get_optimizer_by_name(name)

    if 'optimizers' in settings and settings['optimizers']:
        settings['optimizers'] = [_resolve_optimizer(info) for info in settings['optimizers']]

    if 'optimizer' in settings and settings['optimizer']:
        settings['optimizer'] = _resolve_optimizer(settings['optimizer'])

    if 'starting_point_generator' in settings and settings['starting_point_generator']:
        cls = getattr(mot.cl_routines.optimizing.random_restart, list(settings['starting_point_generator'].keys())[0])
        settings['starting_point_generator'] = cls(**list(settings['starting_point_generator'].values())[0])

    return optimizer(**settings)


def get_general_optimizer_name():
    """Get the name of the currently configured general optimizer

    Returns:
        str: the name of the currently configured optimizer
    """
    return _config['optimization']['general']['name']


def get_general_optimizer_settings():
    """Get the settings of the currently configured general optimizer

    Returns:
        dict: the settings of the currently configured optimizer
    """
    return _config['optimization']['general']['settings']


def get_sampler():
    """Load the sampler from the configuration.

    Returns:
        Sampler: the configured sampler for use in MDT
    """
    sampler = get_sampler_by_name(_config['sampling']['general']['name'])
    return sampler(**_config['sampling']['general']['settings'])


def use_automatic_generated_cascades():
    """Check if we want to use the automatic cascade generation in MDT.

    Returns:
        boolean: True if we want to use the automatic cascades, False otherwise
    """
    return _config['auto_generate_cascade_models']['enabled']


def get_automatic_generated_cascades_excluded():
    """Get the information about the model names excluded from automatic cascade generation.

    Returns:
        list: the names of the composite models we want to exclude from automatic cascade generation
    """
    return _config['auto_generate_cascade_models']['excluded']


def get_model_config(model_names, config):
    """Get from the given dictionary the config for the given model.

    This tries to find the best match between the given config items (by key) and the given model list. For example
    if model_names is ['BallStick', 'S0'] and we have the following config dict:

    .. code-block:: python

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
            recursive list of cascades leading to the composite model we want to get the config for.
        config (dict): the config items with as keys either a composite model regex for a name or a list of regex for
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


@contextmanager
def config_context(config_action):
    """Creates a temporary configuration context with the given config action.

    This will temporarily alter the given configuration keys to the given values. After the context is executed
    the configuration will revert to the original settings.

    Example usage:

    .. code-block:: python

        config = '''
            optimization:
                general:
                    name: 'NMSimplex'
                    settings:
                        patience: 10
        '''
        with mdt.config_context(mdt.configuration.YamlStringAction(config)):
            mdt.fit_model(...)

        This loads the configuration from a YAML string and uses that configuration as the context.

    Args:
        config_action (ConfigAction or str): the configuration action to apply. If a string is given we will
            use it using the YamlStringAction config action.
    """
    if isinstance(config_action, string_types):
        config_action = YamlStringAction(config_action)

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
        load_from_yaml(self._yaml_str)


class SetGeneralSampler(SimpleConfigAction):

    def __init__(self, sampler_name, settings=None):
        super(SetGeneralSampler, self).__init__()
        self._sampler_name = sampler_name
        self._settings = settings or {}

    def _apply(self):
        SampleSettingsLoader().load({'general': {'name': self._sampler_name,
                                                 'settings': self._settings}})


class SetGeneralOptimizer(SimpleConfigAction):

    def __init__(self, optimizer_name, settings=None):
        super(SetGeneralOptimizer, self).__init__()
        self._optimizer_name = optimizer_name
        self._settings = settings or {}

    @classmethod
    def from_object(self, optimizer):
        return SetGeneralOptimizer(optimizer.__class__.__name__, optimizer.optimizer_settings)

    def _apply(self):
        OptimizationSettingsLoader().load({'general': {'name': self._optimizer_name,
                                                       'settings': self._settings}})


"""Load the default configuration, and if possible, the users configuration."""
load_builtin()
try:
    load_user_home()
except IOError:
    pass
