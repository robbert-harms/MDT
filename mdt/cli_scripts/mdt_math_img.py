#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Evaluate an expression on a set of images.

This is meant to quickly convert/combine one or two maps with a mathematical expression.
The expression can be any valid python expression.

The input list of images are loaded as numpy arrays and stored in the array 'input' and 'i'.
Next, the expression is evaluated using the input images and the result is stored in the indicated file.

In the expression you can either use the arrays 'input' or 'i' with linear indices, or/and you can use alphabetic
characters for each image. For example, if you have specified 2 input images
you can address them as:

    - input[0] or i[0] or a
    - input[1] or i[1] or b

This linear alphabetic indexing works with every alphabetic character except for the 'i' since that
one is reserved for the array.

The module numpy is available under 'np' and some functions of MDT under 'mdt'.
This allows expressions like::

    np.mean(np.concatenate(i, axis=3), axis=3)

to get the mean value per voxel of all the input images.

It is possible to change the mode of evaluation from single expression to a more complex python
statement using the switch --as-statement (the default is --as-expression). In a statement
more complex python commands are allowed. In statement mode you must explicitly output the
results using 'return'. (Basically it wraps your command in a function, of which the output is
used as expression value).

If no output file is specified and the output is of dimension 2 or lower we print the output directly
to the console.
"""
import argparse
import glob
import os

import numpy as np

import mdt
from argcomplete.completers import FilesCompleter
import textwrap

from mdt.shell_utils import BasicShellApplication
from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MathImg(BasicShellApplication):

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-math-img fiso.nii ficvf.nii '(1-input[0]) * i[1]' -o Wic.w.nii.gz
            mdt-math-img fiso.nii ficvf.nii '(1-a) * b' -o Wic.w.nii.gz
            mdt-math-img *.nii.gz 'np.mean(np.concatenate(i, axis=3), axis=3)' -o output.nii.gz
            mdt-math-img FA.nii.gz 'np.mean(a)'
            mdt-math-img FA.nii white_matter_mask.nii 'np.mean(mdt.create_roi(a, b))'
            mdt-math-img images*.nii.gz mask.nii 'list(map(lambda f: np.mean(mdt.create_roi(f, i[-1])), i[0:-1]))'
            mdt-math-img FA.nii.gz
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('input_files', metavar='input_files', nargs="+", type=str,
                            help="The input images to use")

        parser.add_argument('expr', metavar='expr', type=str,
                            help="The expression/statement to evaluate.")

        parser.add_argument('-e', '--as-expression', dest='as_expression', action='store_true',
                            help="Evaluates the given string as an expression (default).")
        parser.add_argument('-s', '--as-statement', dest='as_expression', action='store_false',
                            help="Evaluates the given string as an statement.")
        parser.set_defaults(as_expression=True)

        parser.add_argument('-o', '--output-file',
                            help='the output file, if not set nothing is written').completer = \
            FilesCompleter(['nii', 'gz', 'hdr', 'img'], directories=False)

        parser.add_argument('-4d', '--input-4d', action='store_true',
                            help='Add a singleton dimension to all input 3d maps to make them 4d, this prevents '
                                 'some broadcast issues.')

        parser.add_argument('--verbose', '-v', action='store_true', help="Verbose, prints runtime information")

        return parser

    def run(self, args, extra_args):
        write_output = args.output_file is not None

        if write_output:
            output_file = os.path.realpath(args.output_file)

        file_names = []
        images = []
        for file in args.input_files:
            globbed = glob.glob(file)

            if globbed:
                for fname in globbed:
                    file_names.append(fname)
                    images.append(mdt.load_nifti(os.path.realpath(fname)).get_data())
            else:
                file_names.append(file)
                images.append(mdt.load_nifti(os.path.realpath(file)).get_data())

        if args.verbose:
            print('')

        if args.input_4d:
            images = self._images_3d_to_4d(images)

        context_dict = {'input': images, 'i': images, 'np': np, 'mdt': mdt}
        alpha_chars = list('abcdefghjklmnopqrstuvwxyz')

        for ind, image in enumerate(images):
            context_dict.update({alpha_chars[ind]: image})

            if args.verbose:
                print('Input {ind} ({alpha}):'.format(ind=ind, alpha=alpha_chars[ind]))
                print('    name: {}'.format(split_image_path(file_names[ind])[1]))
                print('    shape: {}'.format(str(image.shape)))

        if args.verbose:
            print('')
            print("Evaluating: '{expr}'".format(expr=args.expr))

        if args.as_expression:
            output = eval(args.expr, context_dict)
        else:
            expr = textwrap.dedent('''
            def mdt_image_math():
                {}
            output = mdt_image_math()
            ''').format(args.expr)
            exec(expr, context_dict)
            output = context_dict['output']

        if args.verbose:
            print('')
            if isinstance(output, np.ndarray):
                print('Output shape: {shape}'.format(shape=str(output.shape)))
            else:
                print('Output is single value')

            print('Output: ')
            print('')
            print(output)
        else:
            if not write_output:
                print(output)

        if args.verbose:
            print('')

        if write_output:
            mdt.write_nifti(output, mdt.load_nifti(file_names[0]).get_header(), output_file)

    def _images_3d_to_4d(self, images):
        return list([image[..., np.newaxis] if len(image.shape) == 3 else image for image in images])


def get_doc_arg_parser():
    return MathImg().get_documentation_arg_parser()


if __name__ == '__main__':
    MathImg().start()
