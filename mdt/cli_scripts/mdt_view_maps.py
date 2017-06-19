#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Launches the MDT maps visualizer."""
import argparse
import os
import textwrap
from argcomplete.completers import FilesCompleter
from mdt.utils import init_user_settings
from mdt import view_maps, write_view_maps_figure
from mdt.visualization.maps.base import SimpleDataInfo
from mdt.shell_utils import BasicShellApplication

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GUI(BasicShellApplication):

    def __init__(self):
        super(GUI, self).__init__()
        init_user_settings(pass_if_exists=True)

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('items', metavar='items', type=str, nargs='*', help='the directory or file(s)',
                            default=None).completer = FilesCompleter()

        parser.add_argument('-c', '--config', type=str,
                            help='Use the given initial configuration').completer = \
            FilesCompleter(['conf'], directories=False)

        parser.add_argument('-m', '--maximize', action='store_true', help="Maximize the shown window")
        parser.add_argument('--to-file', type=str, help="If set export the figure to the given filename")

        parser.add_argument('--width', type=int, help="The width of the output file when --to-file is set")
        parser.add_argument('--height', type=int, help="The height of the output file when --to-file is set")
        parser.add_argument('--dpi', type=int, help="The dpi of the output file when --to-file is set")

        return parser

    def run(self, args, extra_args):
        if args.items:
            items = []
            for path in args.items:
                items.append(os.path.realpath(path))
            data = SimpleDataInfo.from_paths(items)
        else:
            data = SimpleDataInfo.from_paths([os.getcwd()])

        to_file = None
        if args.to_file:
            to_file = os.path.realpath(args.to_file)

        config = None
        if args.config:
            filename = os.path.realpath(args.config)
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    config = f.read()

        figure_options = {}
        if to_file:
            if args.width:
                figure_options.update({'width': args.width})
            if args.height:
                figure_options.update({'height': args.height})
            if args.dpi:
                figure_options.update({'dpi': args.dpi})

            write_view_maps_figure(data, to_file, config, figure_options=figure_options)

        else:
            view_maps(data, config, show_maximized=args.maximize)


def get_doc_arg_parser():
    return GUI().get_documentation_arg_parser()


if __name__ == '__main__':
    GUI().start()
