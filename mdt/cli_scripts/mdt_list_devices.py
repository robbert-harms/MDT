#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""This script prints information about the available devices on your computer."""
import argparse
import textwrap
import mdt
from mdt.shell_utils import BasicShellApplication
from mot import cl_environments

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ListDevices(BasicShellApplication):

    def _get_arg_parser(self):
        description = textwrap.dedent(__doc__)
        description += mdt.shell_utils.get_citation_message()

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-l', '--long', action='store_true', help='print all info about the devices')
        return parser

    def run(self, args):
        mdt.init_user_settings(pass_if_exists=True)

        for ind, env in enumerate(cl_environments.CLEnvironmentFactory.smart_device_selection()):
            print('Device {}:'.format(ind))
            if args.long:
                print(repr(env))
            else:
                print(str(env))


if __name__ == '__main__':
    ListDevices().start()
