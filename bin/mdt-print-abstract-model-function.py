#!/usr/bin/env python
import argparse
import sys
import mdt
import mdt.utils
from mdt.cascade_model import CascadeModelInterface

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


if __name__ == '__main__':
    def get_arg_parser():
        description = "This script prints the abstract model function for any of the (non-cascade) models in MDT.\n\n" \
                      "For example, to print the abstract 'BallStick' model function run: \n" \
                      "\tmdt-print-abstract-model-function BallStick\n"

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('model', metavar='model', type=str, help='the model to print')
        return parser

    parser = get_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    model = mdt.get_model(model_name)

    if isinstance(model, CascadeModelInterface):
        print('Printing an abstract model function is not supported for cascade models.')
        sys.exit(2)

    print(model.get_abstract_model_function())