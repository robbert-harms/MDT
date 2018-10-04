#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Print some basic information about an image file."""
import warnings
import argparse
import os
from mdt.lib.nifti import load_nifti, nifti_filepath_resolution
import textwrap

from mdt.lib.shell_utils import BasicShellApplication

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class InfoImg(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-info-img my_img.nii
            mdt-info-img *.nii
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('images', metavar='images', nargs="+", type=str,
                            help="The input images")

        return parser

    def run(self, args, extra_args):
        for image in args.images:
            image_path = os.path.realpath(image)

            try:
                image_path = nifti_filepath_resolution(image_path)
                img = load_nifti(image_path)
                header = img.header
                print('{}'.format(image))
                self.print_info(header)
                print('')

            except ValueError:
                warnings.warn('Could not load image "{}"'.format(image_path))

    def print_info(self, header):
        row_format = "{:<15}{}"
        print(row_format.format('data_type', str(header.get_data_dtype()).upper()))
        print(row_format.format('nmr_dim', len(header.get_data_shape())))

        for ind, el in enumerate(header.get_data_shape()):
            print(row_format.format('dim{}'.format(ind), el))

        for ind, el in enumerate(header.get_zooms()):
            print(row_format.format('pixdim{}'.format(ind), el))


def get_doc_arg_parser():
    return InfoImg().get_documentation_arg_parser()


if __name__ == '__main__':
    InfoImg().start()
