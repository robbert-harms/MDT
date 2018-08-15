#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Generate the bval and bvec file from a protocol file."""
import argparse
import os
import mdt
from argcomplete.completers import FilesCompleter
import textwrap
import mdt.protocols
from mdt.lib.shell_utils import BasicShellApplication
from mdt.protocols import write_bvec_bval

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CreateBvecBval(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-create-bvec-bval my_protocol.prtcl
            mdt-create-bvec-bval my_protocol.prtcl bvec_name.bvec bval_name.bval
        ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('protocol', help='the protocol file').completer = FilesCompleter()
        parser.add_argument('bvec', help="the output bvec file", nargs='?', default=None).completer = FilesCompleter()
        parser.add_argument('bval', help="the output bvec file", nargs='?', default=None).completer = FilesCompleter()

        return parser

    def run(self, args, extra_args):
        protocol_base = os.path.join(os.path.dirname(os.path.realpath(args.protocol)),
                                     os.path.splitext(os.path.basename(args.protocol))[0])

        if args.bvec:
            bvec = os.path.realpath(args.bvec)
        else:
            bvec = protocol_base + '.bvec'

        if args.bval:
            bval = os.path.realpath(args.bval)
        else:
            bval = protocol_base + '.bval'

        write_bvec_bval(mdt.load_protocol(os.path.realpath(args.protocol)), bvec, bval)


def get_doc_arg_parser():
    return CreateBvecBval().get_documentation_arg_parser()


if __name__ == '__main__':
    CreateBvecBval().start()
