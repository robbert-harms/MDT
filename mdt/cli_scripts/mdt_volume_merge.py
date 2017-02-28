#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Merge a list of volume images on the 4th dimension. Writes the result to a single image.

Please note that by default this will sort the list of volume names based on a natural key sort. This is
the most convenient option in the case of globbing files. You can disable this behaviour
using the flag --no-sort.
"""
import argparse
import glob
import os
from argcomplete.completers import FilesCompleter
import textwrap

from mdt.utils import volume_merge
from mdt.shell_utils import BasicShellApplication, get_argparse_extension_checker

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class VolumeMerge(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-volume-merge merged.nii.gz *.nii.gz
            mdt-volume-merge --no-sort merged.nii.gz *.nii.gz
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('-o', '--output_file', required=True,
                            action=get_argparse_extension_checker(['.nii', '.nii.gz', '.hdr', '.img']),
                            help='the filename of the output file').completer = \
            FilesCompleter(['nii', 'gz', 'hdr', 'img'], directories=False)

        parser.add_argument("input_files", metavar="input_files", nargs="+", type=str, help="The input images to merge")

        parser.add_argument('--sort', dest='sort', action='store_true',
                            help="Sort the input images using a natural sort (default)")
        parser.add_argument('--no-sort', dest='sort', action='store_false',
                            help="Do not sort the input images")
        parser.set_defaults(sort=True)

        parser.add_argument('--no-merge-order-file', dest='no_merge_order_file', action='store_true',
                            help="Do not write the merge order file")

        return parser

    def run(self, args, extra_args):
        output_file = os.path.realpath(args.output_file)

        if os.path.isfile(output_file):
            os.remove(output_file)

        file_names = []
        for file in args.input_files:
            file_names.extend(glob.glob(file))

        concatenated_names = volume_merge(file_names, output_file, sort=args.sort)

        if not args.no_merge_order_file:
            info_output_file = os.path.splitext(output_file)[0].replace('.nii', '') + '_merge_order.txt'

            if os.path.isfile(info_output_file):
                os.remove(info_output_file)

            with open(info_output_file, 'w') as f:
                f.write('Files merged in this order:\n')
                for name in concatenated_names:
                    f.write(name + '\n')


def get_doc_arg_parser():
    return VolumeMerge().get_documentation_arg_parser()


if __name__ == '__main__':
    VolumeMerge().start()
