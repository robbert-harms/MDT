#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Evaluate an expression on a protocol.

This is meant to quickly change a protocol using mathematical expressions. The expressions
can be any valid python string separated if needed with the semicolon (;).

The columns of the input protocol are loaded and stored as arrays with as variable names the names of the
columns. Next, the expression is evaluated on those columns and the result is stored in the indicated file.

An additional function "rm(<column_name>)" is also available with wich you can remove columns from
the protocol, and a function "add(<column_name>, <value>)" is available to add columns. When adding
a column, the value can either be a scalar or a vector.

Additionally the numpy library is available with prefix 'np.'.
"""
import argparse
import os
import numpy as np
import mdt
from argcomplete.completers import FilesCompleter
import textwrap
from mdt.shell_utils import BasicShellApplication

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MathProtocol(BasicShellApplication):

    def _get_arg_parser(self):
        description = textwrap.dedent(__doc__)
        description += self._get_citation_message()

        epilog = textwrap.dedent("""
            Examples of use:
                mdt-math-protocol protocol.prtcl 'G *= 1e-3'
                mdt-math-protocol protocol.prtcl 'G *= 1e-3; TR /= 1000; TE /= 1000'
                mdt-math-protocol protocol.prtcl "rm('G')"
                mdt-math-protocol protocol.prtcl "add('TE', 50e-3)"
        """)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_protocol', metavar='input_protocol', type=str,
                            help="The input protocol")

        parser.add_argument('expr', metavar='expr', type=str,
                            help="The expression to evaluate.")

        parser.add_argument('-o', '--output_file',
                            help='the output protocol, defaults to the input protocol.').completer = FilesCompleter()

        return parser

    def run(self, args):
        if args.output_file is not None:
            output_file = os.path.realpath(args.output_file)
        else:
            output_file = os.path.realpath(args.input_protocol)

        protocol = mdt.load_protocol(os.path.realpath(args.input_protocol))
        context_dict = {name: protocol.get_column(name) for name in protocol.column_names}

        def rm(column_name):
            protocol.remove_column(column_name)
            del context_dict[column_name]

        def add(column_name, value):
            protocol.add_column(column_name, value)
            context_dict[column_name] = value

        exec(args.expr, {'np': np, 'rm': rm, 'add': add}, context_dict)

        for name, value in context_dict.items():
            protocol.update_column(name, value)

        mdt.write_protocol(protocol, output_file)


if __name__ == '__main__':
    MathProtocol().start()
