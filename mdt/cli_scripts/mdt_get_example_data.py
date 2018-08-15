#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Get the MDT example data that is accompanying the installation.

This will write the MDT example data (b1k_b2k and b6k datasets) to the indicated directory. You can use this data for
testing MDT on your computer. These example datasets are contained within the MDT package and as such are available
on every machine with MDT installed.
"""
import argparse
import os
import mdt
from argcomplete.completers import FilesCompleter
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GetExampleData(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-get-example-data
            mdt-get-example-data .
        ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('dir', metavar='dir', type=str, nargs='?', help='the output directory',
                            default=None).completer = FilesCompleter()
        return parser

    def run(self, args, extra_args):
        if args.dir:
            output_dir = os.path.realpath(args.dir)
        else:
            output_dir = os.getcwd()

        mdt.get_example_data(output_dir)


def get_doc_arg_parser():
    return GetExampleData().get_documentation_arg_parser()


if __name__ == '__main__':
    GetExampleData().start()
