import os
from os.path import expanduser
import collections
import yaml
from pkg_resources import resource_stream

__author__ = 'Robbert Harms'
__date__ = "2015-06-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

"""The config dictionary."""
config = {}


def load_inbuilt():
    """Load the config file from the skeleton in mdt/data/mdt.conf"""
    with resource_stream('mdt', 'data/mdt.conf') as f:
        load_from_yaml(f.read())


def load_user_home():
    """Load the config file from user home directory"""
    config_file = os.path.join(expanduser("~"), '.mdt', 'mdt.conf')
    if os.path.isfile(config_file):
        with open(config_file) as f:
            load_from_yaml(f.read())


def load_specific(file_name):
    """Can be called by the application to load the config from a specific file.

    This supposes that the given file contains a YAML structured file. That file is read and parsed by the function
    load_from_yaml().

    Please note that the last loading function called overwrites the values of the previous config loads.

    Args:
        file_name (str): The name of the file to load.
    """
    with open(file_name) as f:
        load_from_yaml(f.read())


def load_from_yaml(yaml_str):
    """Can be called to load the config from a yaml string.

    Please note that the last loading function called overwrites the values of the previous config loads.

    Args:
        yaml_str (str): The string containing the YAML config to parse.
    """
    d = yaml.load(yaml_str)
    if d is not None and isinstance(d, dict):
        load_from_dict(d)


def load_from_dict(config_dict):
    """Load configuration from dict.

    Please note that the last loading function called overwrites the values of the previous config loads.

    Args:
        config (dict): a dictionary with configuration options that will overwrite the current configuration.
    """
    _update_dict_recursive(config, config_dict)


def _update_dict_recursive(d, u):
    """Updates in place"""
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = _update_dict_recursive(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


load_inbuilt()
load_user_home()
