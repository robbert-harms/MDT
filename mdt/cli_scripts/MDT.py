#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Launches the MDT GUI, similar to the mdt-gui command"""
from mdt.cli_scripts.mdt_gui import GUI

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GUI_Shortcut(GUI):
    pass


def get_doc_arg_parser():
    return GUI_Shortcut().get_documentation_arg_parser()


if __name__ == '__main__':
    GUI_Shortcut().start()
