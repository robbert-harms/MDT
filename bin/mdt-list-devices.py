#!/usr/bin/env python
import argparse
from mot import cl_environments

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

def get_arg_parser():
    description = "This script prints information about the available devices on your computer.\n"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    return parser

parser = get_arg_parser()
args = parser.parse_args()

for env in cl_environments.CLEnvironmentFactory.all_devices():
    print(repr(env))
    print("\n")

