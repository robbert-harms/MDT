#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Launches the MDT maps visualizer."""
import argparse
import os
import textwrap

from argcomplete.completers import FilesCompleter

from mdt import init_user_settings, view_maps
from mdt.visualization.maps.base import DataInfo
from mdt.shell_utils import BasicShellApplication, get_citation_message

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GUI(BasicShellApplication):

    def __init__(self):
        init_user_settings(pass_if_exists=True)

    def _get_arg_parser(self):
        description = textwrap.dedent(__doc__)
        description += get_citation_message()

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('dir', metavar='dir', type=str, nargs='?', help='the directory to use',
                            default=None).completer = FilesCompleter()

        parser.add_argument('-c', '--config', type=str,
                            help='Use the given initial configuration').completer = \
            FilesCompleter(['conf'], directories=False)

        parser.add_argument('-m', '--maximize', action='store_true', help="Maximize the shown window")
        parser.add_argument('--to-file', type=str, help="If set export the figure to the given filename")

        return parser

    def run(self, args):
        if args.dir:
            data = DataInfo.from_dir(os.path.realpath(args.dir))
        else:
            data = DataInfo.from_dir(os.getcwd())

        to_file = None
        if args.to_file:
            to_file = os.path.realpath(args.to_file)

        config = None
        if args.config:
            filename = os.path.realpath(args.config)
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    config = f.read()

        view_maps(data, config, show_maximized=args.maximize, to_file=to_file)

if __name__ == '__main__':
    GUI().start()
