"""Contains the runtime configuration of MDT.

This consists of two parts, functions to get the current runtime settings and configuration actions to update these
settings. To set a new configuration, create a new :py:class:`ConfigAction` and use this within a context environment
using :py:func:`config_context`. Example:

.. code-block:: python

    from mdt.configuration import YamlStringAction, config_context

    config = '''
        optimization:
            general:
                name: 'Powell'
                settings:
                    patience: 2
    '''
    with mdt.config_context(YamlStringAction(config)):
        mdt.fit_model(...)
"""
import os
import re
from copy import deepcopy

import yaml
from contextlib import contextmanager
from pkg_resources import resource_stream

import mot.configuration

from mdt.__version__ import __version__

__author__ = 'Robbert Harms'
__date__ = "2015-06-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

""" The current configuration """
_config = {}


def get_config_option(option_name):
    """Get the current configuration option for the given option name.

    Args:
        option_name (list of str or str): the name of the option, or a path to the option.

    Returns:
        object: the raw configuration value defined for that option
    """
    if isinstance(option_name, str):
        return _config[option_name]
    else:
        config = _config
        for el in option_name[:-1]:
            config = config[el]
        return config[option_name[-1]]


def set_config_option(option_name, value):
    """Set the current configuration option for the given option name.

    This will overwrite the current configuration for that option with the given value. Be careful, this will change
    the global configuration value.

    Provided values should be objects and not YAML strings. For updating the configuration with YAML strings, please use
    the function :func:`load_from_yaml`.

    Args:
        option_name (list of str or str): the name of the option, or a path to the option.
        value : the object to set for that option

    Returns:
        object: the raw configuration value defined for that option
    """
    if isinstance(option_name, str):
        _config[option_name] = value
    else:
        config = _config
        for el in option_name[:-1]:
            config = config[el]
        config[option_name[-1]] = value


def get_config_dir():
    """Get the location of the components.

    Return:
        str: the path to the components
    """
    return os.path.join(os.path.expanduser("~"), '.mdt', __version__)


def _config_insert(keys, value):
    """Insert the given value in the given key.

    This will create all layers of the dictionary if needed.

    Args:
        keys (list of str): the position of the input value
        value : the value to put at the position of the key.
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

    Also, please note that this will change the global configuration, i.e. this is a persistent change. If you do not
    want a persistent state change, consider using :func:`~mdt.configuration.config_context` instead.

    Args:
        file_name (str): The name of the file to use.
    """
    with open(file_name) as f:
        load_from_yaml(f.read())


def load_from_yaml(yaml_str):
    """Can be called to use configuration options from a YAML string.

    This will update the current configuration with the new options.

    Please note that this will change the global configuration, i.e. this is a persistent change. If you do not
    want a persistent state change, consider using :func:`~mdt.configuration.config_context` instead.

    Args:
        yaml_str (str): The string containing the YAML config to parse.
    """
    config_dict = yaml.safe_load(yaml_str) or {}
    load_from_dict(config_dict)


def load_from_dict(config_dict):
    """Load configuration options from a given dictionary.

    Please note that this will change the global configuration, i.e. this is a persistent change. If you do not
    want a persistent state change, consider using :func:`~mdt.configuration.config_context` instead.

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


class ConfigSectionLoader:

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
                _config_insert(['output_format', item, 'gzip'], bool(options['gzip']))


class LoggingLoader(ConfigSectionLoader):
    """Loader for the top level key logging. """

    def load(self, value):
        ensure_exists(['logging', 'info_dict'])
        if 'info_dict' in value:
            self._load_info_dict(value['info_dict'])

    def _load_info_dict(self, info_dict):
        for item in ['version', 'disable_existing_loggers', 'formatters', 'handlers', 'loggers', 'root']:
            if item in info_dict:
                _config_insert(['logging', 'info_dict', item], info_dict[item])


class OptimizationSettingsLoader(ConfigSectionLoader):
    """Loads the optimization section"""

    def load(self, value):
        ensure_exists(['optimization', 'general'])
        ensure_exists(['optimization', 'model_specific'])

        if 'general' in value:
            _config_insert(['optimization', 'general'], value['general'])

        if 'model_specific' in value:
            for key, sub_value in value['model_specific'].items():
                _config_insert(['optimization', 'model_specific', key], sub_value)


class SampleSettingsLoader(ConfigSectionLoader):
    """Loads the sample section"""

    def load(self, value):
        ensure_exists(['sampling', 'general'])

        settings = value.get('settings', {})
        settings['nmr_samples'] = settings.get('nmr_samples', 10000)
        settings['burnin'] = settings.get('burnin', 0)
        settings['thinning'] = settings.get('thinning', 1)
        _config_insert(['sampling', 'general', 'settings'], settings)


class ProcessingStrategySectionLoader(ConfigSectionLoader):
    """Loads the config section processing_strategies"""

    def load(self, value):
        if 'optimization' in value:
            _config_insert(['processing_strategies', 'optimization'], value['optimization'])
        if 'sampling' in value:
            _config_insert(['processing_strategies', 'sampling'], value['sampling'])


class TmpResultsDirSectionLoader(ConfigSectionLoader):
    """Load the section tmp_results_dir"""

    def load(self, value):
        _config_insert(['tmp_results_dir'], value)


class ActivePostProcessingLoader(ConfigSectionLoader):
    """Load the default settings for the post sample calculations."""

    def load(self, value):
        sampling = value.get('sampling', {})
        sampling['univariate_ess'] = sampling.get('univariate_ess', False)
        sampling['multivariate_ess'] = sampling.get('multivariate_ess', False)
        sampling['maximum_likelihood'] = sampling.get('maximum_likelihood', False)
        sampling['maximum_a_posteriori'] = sampling.get('maximum_a_posteriori', False)
        sampling['model_defined_maps'] = sampling.get('model_defined_maps', True)
        sampling['univariate_normal'] = sampling.get('univariate_normal', True)
        sampling['average_acceptance_rate'] = sampling.get('average_acceptance_rate', False)

        optimization = value.get('optimization', {})
        optimization['uncertainties'] = optimization.get('uncertainties', True)
        optimization['store_covariances'] = optimization.get('store_covariances', True)

        _config_insert(['active_post_processing', 'optimization'], optimization)
        _config_insert(['active_post_processing', 'sampling'], sampling)


class AutomaticCascadeModels(ConfigSectionLoader):
    """Load the automatic cascade model settings."""

    def load(self, value):
        _config_insert(['auto_generate_cascade_models', 'enabled'], value.get('enabled', True))
        _config_insert(['auto_generate_cascade_models', 'excluded'], value.get('excluded', []))


class RuntimeSettingsLoader(ConfigSectionLoader):

    def load(self, value):
        if 'cl_device_ind' in value:
            if value['cl_device_ind'] is not None:
                from mdt.utils import get_cl_devices
                devices = get_cl_devices(value['cl_device_ind'])

                if devices:
                    mot.configuration.set_cl_environments(devices)

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

    if section == 'runtime_settings':
        return RuntimeSettingsLoader()

    if section == 'auto_generate_cascade_models':
        return AutomaticCascadeModels()

    if section == 'active_post_processing':
        return ActivePostProcessingLoader()

    raise ValueError('Could not find a suitable configuration loader for the section {}.'.format(section))


def gzip_optimization_results():
    """Check if we should write the volume maps from the optimization gzipped or not.

    Returns:
        boolean: True if the results of optimization computations should be gzipped, False otherwise.
    """
    return _config['output_format']['optimization']['gzip']


def gzip_sampling_results():
    """Check if we should write the volume maps from the sample gzipped or not.

    Returns:
        boolean: True if the results of sample computations should be gzipped, False otherwise.
    """
    return _config['output_format']['sampling']['gzip']


def get_tmp_results_dir():
    """Get the default tmp results directory.

    This is the default directory for saving temporary computation results. Set to None to disable this and
    use the model directory.

    Returns:
        str or None: the tmp results dir to use during optimization and sample
    """
    return _config['tmp_results_dir']


def get_active_post_processing():
    """Get the overview of active post processing switches.

    Returns:
        dict: a dictionary holding two dictionaries, one called 'optimization' and one called 'sampling'.
            Both these dictionaries hold keys of elements to add to the respective post processing phase.
    """
    return deepcopy(_config['active_post_processing'])


def get_processing_strategy(processing_type, *args, **kwargs):
    """Get the correct processing strategy for the given model.

    Args:
        processing_type (str): 'optimization', 'sampling' or any other of the
            processing_strategies defined in the config
        model_names (list of str): the list of model names (the full recursive cascade of model names)
        **kwargs: passed to the constructor of the loaded processing strategy.

    Returns:
        ModelProcessingStrategy: the processing strategy to use for this model
    """
    from mdt.lib.processing_strategies import VoxelRange
    options = _config['processing_strategies'].get(processing_type, {}) or {}
    options.update(kwargs)
    return VoxelRange(*args, **options)


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
    return _config['optimization']['general']['name']


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
        return info_dict['name']
    else:
        return get_general_optimizer()


def get_general_optimizer_name():
    """Get the name of the currently configured general optimizer

    Returns:
        str: the name of the currently configured optimizer
    """
    return _config['optimization']['general']['name']


def get_general_optimizer_options():
    """Get the settings of the currently configured general optimizer

    Returns:
        dict: the settings of the currently configured optimizer
    """
    return _config['optimization']['general']['options']


def get_general_sampling_settings():
    """Get the general sample settings.

    Returns:
        Sampler: the configured sampler for use in MDT
    """
    return _config['sampling']['general']['settings']


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
        if isinstance(model_names, str):
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
                    name: 'Nelder-Mead'
                    options:
                        patience: 10
        '''
        with mdt.config_context(mdt.configuration.YamlStringAction(config)):
            mdt.fit_model(...)


    or, equivalently::

        config = '''
            ...
        '''
        with mdt.config_context(config):
            ...

    This loads the configuration from a YAML string and uses that configuration as the context.

    Args:
        config_action (mdt.configuration.ConfigAction or str): the configuration action to apply.
            If a string is given we will use it using the YamlStringAction config action.
    """
    if isinstance(config_action, str):
        config_action = YamlStringAction(config_action)

    config_action.apply()
    yield
    config_action.unapply()


class ConfigAction:

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
        super().__init__()
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
        super().__init__()
        self._yaml_str = yaml_str

    def _apply(self):
        load_from_yaml(self._yaml_str)


class SetGeneralSampler(SimpleConfigAction):

    def __init__(self, sampler_name, settings=None):
        super().__init__()
        self._sampler_name = sampler_name
        self._settings = settings or {}

    def _apply(self):
        SampleSettingsLoader().load({'general': {'name': self._sampler_name,
                                                 'settings': self._settings}})


class SetGeneralOptimizer(SimpleConfigAction):

    def __init__(self, optimizer_name, settings=None):
        super().__init__()
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
