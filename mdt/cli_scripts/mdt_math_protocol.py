#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Evaluate an expression on a protocol.

This is meant to quickly change a protocol using mathematical expressions. The expressions
can be any valid python string separated if needed with the semicolon (;).

The columns of the input protocol are loaded and stored as arrays with as variable names the names of the
columns. Next, the expression is evaluated on those columns and the result is stored in the indicated file.

Columns can easily be removed with the python 'del' command. New columns can easily be added by assignment.When adding
a column, the value can either be a scalar or a vector.

Additionally, the numpy library is available with prefix 'np.'.
"""
import argparse
import os
import numpy as np
import mdt
from argcomplete.completers import FilesCompleter
import textwrap

from mdt.protocols import Protocol
from mdt.shell_utils import BasicShellApplication
from mot.utils import is_scalar

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MathProtocol(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-math-protocol protocol.prtcl 'G *= 1e-3' -o new_protocol.prtcl
            mdt-math-protocol p.prtcl 'G *= 1e-3; TR /= 1000; TE /= 1000'
            mdt-math-protocol p.prtcl 'del(G)'
            mdt-math-protocol p.prtcl 'TE  = 50e-3'
            mdt-math-protocol p.prtcl -a Delta.txt 'Delta = files[0]'
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_protocol', metavar='input_protocol', type=str,
                            help="The input protocol")

        parser.add_argument('expr', metavar='expr', type=str,
                            help="The expression to evaluate.")

        parser.add_argument('-o', '--output_file',
                            help='the output protocol, defaults to the input protocol.').completer = FilesCompleter()

        parser.add_argument('-a', '--additional-file', type=str, action='append', dest='additional_files',
                            help='additional file to load to be used for columns, placed in \'files\' list by index')

        return parser

    def run(self, args, extra_args):
        if args.output_file is not None:
            output_file = os.path.realpath(args.output_file)
        else:
            output_file = os.path.realpath(args.input_protocol)

        additional_files = []
        if args.additional_files:
            for file in args.additional_files:
                additional_files.append(np.genfromtxt(file))

        protocol = mdt.load_protocol(os.path.realpath(args.input_protocol))
        context_dict = {name: protocol.get_column(name) for name in protocol.column_names}

        exec(args.expr, {'np': np, 'files': additional_files}, context_dict)

        for key in context_dict:
            if is_scalar(context_dict[key]):
                context_dict[key] = np.ones(protocol.length) * context_dict[key]

        protocol = Protocol(context_dict)
        mdt.write_protocol(protocol, output_file)


def get_doc_arg_parser():
    return MathProtocol().get_documentation_arg_parser()


if __name__ == '__main__':
    MathProtocol().start()
