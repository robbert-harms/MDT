import os
import yaml
from pkg_resources import resource_stream

__author__ = 'Robbert Harms'
__date__ = "2015-06-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

"""The config dictionary."""
config = {}


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


config = load_builtin()
try:
    config = load_user_home()
except ValueError:
    pass
