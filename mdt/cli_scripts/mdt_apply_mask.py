#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Mask the background using the given mask.

This function multiplies a given volume (or list of volumes) with a binary mask.

Please note that this changes the input files (changes are in-place).
"""
import argparse
import glob
import os
import mdt
from argcomplete.completers import FilesCompleter
from mdt.shell_utils import BasicShellApplication
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ApplyMask(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-apply-mask data.nii.gz -m roi_mask_0_50.nii.gz
            mdt-apply-mask *.nii.gz -m my_mask.nii.gz
        ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_files', metavar='input_files', nargs="+", type=str,
                            help="The input images to use")

        parser.add_argument('-m', '--mask', required=True,
                            action=mdt.shell_utils.get_argparse_extension_checker(['.nii', '.nii.gz', '.hdr', '.img']),
                            help='the (brain) mask to use').completer = FilesCompleter(['nii', 'gz', 'hdr', 'img'],
                                                                                       directories=False)
        return parser

    def run(self, args, extra_args):
        mask = mdt.load_brain_mask(os.path.realpath(args.mask))

        file_names = []
        for file in args.input_files:
            file_names.extend(glob.glob(file))

        for file in file_names:
            mdt.apply_mask_to_file(file, mask)


def get_doc_arg_parser():
    return ApplyMask().get_documentation_arg_parser()


if __name__ == '__main__':
    ApplyMask().start()
