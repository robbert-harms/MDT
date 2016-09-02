#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import os
import textwrap

from argcomplete.completers import FilesCompleter

from mdt import init_user_settings
from mdt.gui.maps_visualizer.main import start_gui
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
        description = textwrap.dedent("""
            Launches the MDT maps visualizer.
        """)
        description += get_citation_message()

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('dir', metavar='dir', type=str, nargs='?', help='the directory to load',
                            default=None).completer = FilesCompleter()
        return parser

    def run(self, args):
        data = None
        if args.dir:
            data = DataInfo.from_dir(os.path.realpath(args.dir))

        start_gui(data)

if __name__ == '__main__':
    GUI().start()
