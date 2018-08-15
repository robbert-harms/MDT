#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Create a single slice mask that only includes the voxels in the selected slice."""
import argparse
import os
import mdt
from argcomplete.completers import FilesCompleter
import textwrap
from mdt.lib.nifti import load_nifti
import mdt.utils
from mdt.lib.shell_utils import BasicShellApplication

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CreateRoiSlice(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-create-roi-slice mask.nii.gz
            mdt-create-roi-slice mask.nii.gz -d 1 -s 50
            mdt-create-roi-slice mask.nii.gz -d 1 -s 50 -o my_roi_1_50.nii.gz
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('mask',
                            action=mdt.lib.shell_utils.get_argparse_extension_checker(['.nii', '.nii.gz', '.hdr', '.img']),
                            help='the mask to select a slice from').completer = \
            FilesCompleter(['nii', 'gz', 'hdr', 'img'], directories=False)

        parser.add_argument('-d', '--dimension', type=int, help="The dimension to index (0, 1, 2, ...). Default is 2.")
        parser.add_argument('-s', '--slice', type=int, help="The slice to use in the selected dimension (0, 1, 2, ...)."
                                                            "Defaults to center of chosen dimension.")

        parser.add_argument('-o', '--output-name',
                            action=mdt.lib.shell_utils.get_argparse_extension_checker(['.nii', '.nii.gz', '.hdr', '.img']),
                            help='the filename of the output file. Default is <mask_name>_<dim>_<slice>.nii.gz').\
            completer = FilesCompleter(['nii', 'gz', 'hdr', 'img'], directories=False)

        return parser

    def run(self, args, extra_args):
        shape = load_nifti(args.mask).shape
        roi_dimension = args.dimension if args.dimension is not None else 2
        if roi_dimension > len(shape)-1 or roi_dimension < 0:
            print('Error: the given mask has only {0} dimensions with slices {1}.'.format(len(shape), shape))
            exit(1)

        roi_slice = args.slice if args.slice is not None else shape[roi_dimension] // 2
        if roi_slice > shape[roi_dimension]-1 or roi_slice < 0:
            print('Error: dimension {0} has only {1} slices.'.format(roi_dimension, shape[roi_dimension]))
            exit(1)

        mask_base_name = os.path.splitext(os.path.realpath(args.mask))[0]
        mask_base_name = mask_base_name.replace('.nii', '')

        if args.output_name:
            output_name = os.path.realpath(args.output_name)
        else:
            output_name = mask_base_name + '_{0}_{1}.nii.gz'.format(roi_dimension, roi_slice)

        mdt.utils.write_slice_roi(os.path.realpath(args.mask), roi_dimension, roi_slice, output_name,
                                  overwrite_if_exists=True)


def get_doc_arg_parser():
    return CreateRoiSlice().get_documentation_arg_parser()


if __name__ == '__main__':
    CreateRoiSlice().start()
