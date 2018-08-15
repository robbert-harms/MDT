#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Mask the background using the given mask.

This function multiplies a given volume (or list of volumes) with a binary mask.
"""
import argparse
import glob
import os
import mdt
from argcomplete.completers import FilesCompleter

from mdt.lib.nifti import nifti_filepath_resolution
from mdt.lib.shell_utils import BasicShellApplication
import textwrap

from mdt.utils import split_image_path

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

        parser.add_argument('mask', help='the (brain) mask to use').completer = \
            FilesCompleter(['nii', 'gz', 'hdr', 'img'], directories=False)

        parser.add_argument('input_files', metavar='input_files', nargs="+", type=str,
                            help="The input images to use")

        parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                            help="Overwrite the original images, if not set we create an output file.")
        parser.set_defaults(overwrite=False)

        return parser

    def run(self, args, extra_args):
        mask = mdt.load_brain_mask(nifti_filepath_resolution(os.path.realpath(args.mask)))

        file_names = []
        for file in args.input_files:
            file_names.extend(glob.glob(file))

        for file in file_names:
            if args.overwrite:
                mdt.apply_mask_to_file(file, mask)
            else:
                folder, basename, ext = split_image_path(nifti_filepath_resolution(os.path.realpath(file)))
                mdt.apply_mask_to_file(file, mask, output_fname=os.path.join(folder, basename + '_masked' + ext))


def get_doc_arg_parser():
    return ApplyMask().get_documentation_arg_parser()


if __name__ == '__main__':
    ApplyMask().start()
