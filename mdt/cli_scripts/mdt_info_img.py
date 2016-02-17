#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
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


class InfoImg(BasicShellApplication):

    def _get_arg_parser(self):
        description = textwrap.dedent("""
            Print some basic information about an image file.
        """)
        description += self._get_citation_message()

        epilog = textwrap.dedent("""
            Examples of use:
                mdt-info-img my_img.nii
        """)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('image',
                            action=mdt.shell_utils.get_argparse_extension_checker(['.nii', '.nii.gz', '.hdr', '.img']),
                            help='the input image').completer = \
            FilesCompleter(['nii', 'gz', 'hdr', 'img'], directories=False)

        return parser

    def run(self, args):
        image = os.path.realpath(args.image)
        img = mdt.load_nifti(image)
        header = img.get_header()
        self.print_info(header)

    def print_info(self, header):
        row_format = "{:<15}{}"
        print(row_format.format('data_type', str(header.get_data_dtype()).upper()))
        print(row_format.format('nmr_dim', len(header.get_data_shape())))

        for ind, el in enumerate(header.get_data_shape()):
            print(row_format.format('dim{}'.format(ind), el))

        for ind, el in enumerate(header.get_zooms()):
            print(row_format.format('pixdim{}'.format(ind), el))


if __name__ == '__main__':
    InfoImg().start()
