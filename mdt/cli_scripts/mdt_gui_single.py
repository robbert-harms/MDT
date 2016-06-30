#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import os
import textwrap
from argcomplete.completers import FilesCompleter
from mdt.shell_utils import BasicShellApplication, get_citation_message

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GUISingle(BasicShellApplication):

    def __init__(self):
        """
        Normally we would load here the user settings. This can not be
        done however since it would load the MOT utils before the GUI is launched. This will give OpenCL errors
        ("RuntimeError: CommandQueue failed: out of host memory") when creating the _logging_update_queue.

        Hence we do not use:
            mdt.init_user_settings(pass_if_exists=True)
        """

    def _get_arg_parser(self):
        description = textwrap.dedent("""
            Launches the MDT single subject Graphical User Interface.
        """)
        description += get_citation_message()

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-d', '--dir', metavar='dir', type=str, help='the base directory for the file choosers',
                            default=None).completer = FilesCompleter()

        parser.add_argument('-tk', dest='tk', action='store_true', help="Launch the TK gui (default)")
        parser.add_argument('-qt', dest='qt', action='store_true', help="Launch the QT gui")

        parser.add_argument('-m', '--view_maps', dest='maps', action='store_true', help="Directly open the tab "
                                                                                   "for viewing maps")

        return parser

    def run(self, args):
        if args.dir:
            cwd = os.path.realpath(args.dir)
        else:
            cwd = os.getcwd()

        if args.qt:
            from mdt.gui.qt_main import start_single_model_gui
        else:
            from mdt.gui.tkgui_main import start_single_model_gui

        action = None
        if args.maps:
            action = 'view_maps'

        start_single_model_gui(cwd, action)


if __name__ == '__main__':
    GUISingle().start()
