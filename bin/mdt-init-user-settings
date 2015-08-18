#!/usr/bin/env python
import argparse
import mdt

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_arg_parser():
    description = "This script is meant to update your home folder with the latest MDT models.\n"

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    return parser

parser = get_arg_parser()
args = parser.parse_args()

mdt.initialize_user_settings()
