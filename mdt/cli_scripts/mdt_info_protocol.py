#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Print some basic information about a protocol."""
import argparse
import os
import mdt
from argcomplete.completers import FilesCompleter
import textwrap

from mdt.shell_utils import BasicShellApplication

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class InfoProtocol(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-info-protocol my_protocol.prtcl
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('protocol',
                            action=mdt.shell_utils.get_argparse_extension_checker(['.prtcl']),
                            help='the protocol file').completer = FilesCompleter(['prtcl'], directories=False)

        return parser

    def run(self, args, extra_args):
        protocol = mdt.load_protocol(os.path.realpath(args.protocol))
        self.print_info(protocol)

    def print_info(self, protocol):
        row_format = "{:<15}{}"

        print(row_format.format('nmr_rows', protocol.length))
        print(row_format.format('nmr_unweighted', len(protocol.get_unweighted_indices())))
        print(row_format.format('nmr_weighted', len(protocol.get_weighted_indices())))
        print(row_format.format('nmr_shells', len(protocol.get_b_values_shells())))

        shells = protocol.get_b_values_shells()
        shells_text = []
        for shell in shells:
            occurrences = protocol.count_occurences('b', shell)
            shells_text.append('{0:0=.3f}e9 ({1})'.format(shell / 1e9, occurrences))
        print(row_format.format('shells', ', '.join(shells_text)))
        print(row_format.format('nmr_columns', protocol.number_of_columns))
        print(row_format.format('columns', ', '.join(protocol.column_names)))


def get_doc_arg_parser():
    return InfoProtocol().get_documentation_arg_parser()


if __name__ == '__main__':
    InfoProtocol().start()
