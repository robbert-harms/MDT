#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""This script prints a list of all the models MDT can find in your home directory."""
import argparse
import mdt
import mdt.utils
from mdt.shell_utils import BasicShellApplication

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ListModels(BasicShellApplication):

    def _get_arg_parser(self):
        description = __doc__
        description += mdt.shell_utils.get_citation_message()

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-l', '--long', action='store_true', help='print the descriptions')
        return parser

    def run(self, args):
        mdt.init_user_settings(pass_if_exists=True)

        meta_info = mdt.get_models_meta_info()
        models = mdt.get_models_list()

        max_model_name = max(map(len, models))

        for model in models:
            if args.long:
                print(('%-' + str(max_model_name + 2) + 's%-s') % (model, meta_info[model]['description']))
            else:
                print(model)


if __name__ == '__main__':
    ListModels().start()
