#!/usr/bin/env python
import argparse
from mdt.gui import tkgui

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_arg_parser():
    description = "Launches the MDT TK Graphical User Interface.\n"
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    return parser

parser = get_arg_parser()
args = parser.parse_args()

window = tkgui.get_window()
window.mainloop()