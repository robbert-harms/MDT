#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""This script is meant to update your home folder with the latest MDT models."""
import argparse
import os
import textwrap
import mdt
from mdt.configuration import get_config_dir
from mdt.shell_utils import BasicShellApplication

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class InitUserSettings(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__ + """

            The location we will write to is: {}
        """.format(os.path.dirname(get_config_dir())))

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('--pass-if-exists', dest='pass_if_exists', action='store_true',
                            help="do nothing if the config dir exists (default)")
        parser.add_argument('--always-overwrite', dest='pass_if_exists', action='store_false',
                            help="always overwrite the config directory with the default settings")
        parser.set_defaults(pass_if_exists=False)

        return parser

    def run(self, args, extra_args):
        mdt.init_user_settings(pass_if_exists=args.pass_if_exists)


def get_doc_arg_parser():
    return InitUserSettings().get_documentation_arg_parser()


if __name__ == '__main__':
    InitUserSettings().start()
