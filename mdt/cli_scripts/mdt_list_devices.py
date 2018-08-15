#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""This script prints information about the available devices on your computer."""
import argparse
import textwrap
from mdt.lib.shell_utils import BasicShellApplication
from mot.lib import cl_environments

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ListDevices(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-l', '--long', action='store_true', help='print all info about the devices')
        return parser

    def run(self, args, extra_args):
        for ind, env in enumerate(cl_environments.CLEnvironmentFactory.smart_device_selection()):
            print('Device {}:'.format(ind))
            if args.long:
                print(repr(env))
            else:
                print(str(env))


def get_doc_arg_parser():
    return ListDevices().get_documentation_arg_parser()


if __name__ == '__main__':
    ListDevices().start()
