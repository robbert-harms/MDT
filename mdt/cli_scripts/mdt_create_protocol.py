#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Create a protocol from a bvec and bval file.

MDT uses a protocol file (with extension .prtcl) to store all the acquisition related values.
This is a column based file which can hold, next to the b-values and gradient directions,
the big Delta, small delta, gradient amplitude G and more of these extra acquisition details.
"""
import argparse
import os

from argcomplete.completers import FilesCompleter
import textwrap
import mdt.protocols
from mdt.lib.shell_utils import BasicShellApplication
from mdt.protocols import load_bvec_bval

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CreateProtocol(BasicShellApplication):

    def __init__(self):
        super().__init__()
        self.parse_unknown_args = True

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-create-protocol data.bvec data.bval
            mdt-create-protocol data.bvec data.bval -o my_protocol.prtcl
            mdt-create-protocol data.bvec data.bval
            mdt-create-protocol data.bvec data.bval --Delta 30 --delta 20
            mdt-create-protocol data.bvec data.bval --sequence-timing-units 's' --Delta 0.03
            mdt-create-protocol data.bvec data.bval --TE ../my_TE_file.txt
           ''')
        epilog = self._format_examples(doc_parser, examples)
        epilog += textwrap.dedent("""

            Additional columns can be specified using the syntax: \"--{column_name} {value}\" structure.
            Please note that these additional values will not be auto-converted from ms to s.
        """)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('bvec', help='the gradient vectors file').completer = FilesCompleter()
        parser.add_argument('bval', help='the gradient b-values').completer = FilesCompleter()
        parser.add_argument('-s', '--bval-scale-factor', type=float,
                            help="We expect the b-values in the output protocol in units of s/m^2. "
                                 "Example use: 1 or 1e6. The default is autodetect.")

        parser.add_argument('-o', '--output_file',
                            help='the output protocol, defaults to "<bvec_name>.prtcl" in the same '
                                 'directory as the bvec file.').completer = FilesCompleter()

        parser.add_argument('--sequence-timing-units', choices=('ms', 's'), default='ms',
                            help="The units of the sequence timings. The default is 'ms' which we will convert to 's'.")

        parser.add_argument('--G',
                            help="The gradient amplitudes in T/m.")

        parser.add_argument('--maxG',
                            help="The maximum gradient amplitude in T/m. This is only useful if we need to guess "
                                 "big Delta and small delta. Default is 0.04 T/m")

        parser.add_argument('--Delta',
                            help="The big Delta to use, either a single number or a file with either a single number "
                                 "or one number per gradient direction.")

        parser.add_argument('--delta',
                            help="The small delta to use, either a single number or a file with either a single number "
                                 "or one number per gradient direction.")

        parser.add_argument('--TE',
                            help="The TE to use, either a single number or a file with either a single number "
                                 "or one number per gradient direction.")

        parser.add_argument('--TR',
                            help="The TR to use, either a single number or a file with either a single number "
                                 "or one number per gradient direction.")

        return parser

    def run(self, args, extra_args):
        bvec = os.path.realpath(args.bvec)
        bval = os.path.realpath(args.bval)

        if args.output_file:
            output_prtcl = os.path.realpath(args.output_file)
        else:
            output_prtcl = os.path.join(os.path.dirname(bvec),
                                        os.path.splitext(os.path.basename(bvec))[0] + '.prtcl')

        if args.bval_scale_factor:
            bval_scale_factor = float(args.bval_scale_factor)
        else:
            bval_scale_factor = 'auto'

        protocol = load_bvec_bval(bvec=bvec, bval=bval, bval_scale=bval_scale_factor)

        if args.G is None and args.maxG is not None:
            if os.path.isfile(str(args.maxG)):
                protocol = protocol.with_added_column_from_file('maxG', os.path.realpath(str(args.maxG)), 1)
            else:
                protocol = protocol.with_new_column('maxG', float(args.maxG))

        if args.Delta is not None:
            protocol = add_sequence_timing_column_to_protocol(protocol, 'Delta', args.Delta, args.sequence_timing_units)
        if args.delta is not None:
            protocol = add_sequence_timing_column_to_protocol(protocol, 'delta', args.delta, args.sequence_timing_units)
        if args.TE is not None:
            protocol = add_sequence_timing_column_to_protocol(protocol, 'TE', args.TE, args.sequence_timing_units)
        if args.TR is not None:
            protocol = add_sequence_timing_column_to_protocol(protocol, 'TR', args.TR, args.sequence_timing_units)
        if args.G is not None:
            protocol = add_column_to_protocol(protocol, 'G', args.G, 1)

        protocol = add_extra_columns(protocol, extra_args)

        mdt.protocols.write_protocol(protocol, output_prtcl)


def add_extra_columns(protocol, extra_args):
    key = None
    for element in extra_args:
        if '=' in element and element.startswith('--'):
            key, value = element[2:].split('=')
            protocol = add_column_to_protocol(protocol, key, value, 1)
        elif element.startswith('--'):
            key = element[2:]
        else:
            protocol = add_column_to_protocol(protocol, key, element, 1)
    return protocol


def add_column_to_protocol(protocol, column, value, mult_factor):
    if value is not None:
        if os.path.isfile(value):
            return protocol.with_added_column_from_file(column, os.path.realpath(value), mult_factor)
        else:
            return protocol.with_new_column(column, float(value) * mult_factor)


def add_sequence_timing_column_to_protocol(protocol, column, value, units):
    mult_factor = 1e-3 if units == 'ms' else 1
    return add_column_to_protocol(protocol, column, value, mult_factor)


def get_doc_arg_parser():
    return CreateProtocol().get_documentation_arg_parser()


if __name__ == '__main__':
    CreateProtocol().start()
