#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Fit one of the models to the given data.

This function can use two kinds of noise standard deviation, a global or a local (voxel wise).
If the argument -n / --noise-std is not set, MDT uses a default automatic noise estimation which
may be either global or local. To use a predefined global noise std please set the argument to a
floating point value. To use a voxel wise noise std, please give it a filename with a map to use.
"""
import argparse
import os
import mdt
from argcomplete.completers import FilesCompleter

from mdt import DMRICascadeModelInterface
from mdt.lib.shell_utils import BasicShellApplication
from mot.lib import cl_environments
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelFit(BasicShellApplication):

    def __init__(self):
        super().__init__()
        self.available_devices = list((ind for ind, env in
                                       enumerate(cl_environments.CLEnvironmentFactory.smart_device_selection())))

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-model-fit BallStick_r1 data.nii.gz data.prtcl roi_mask_0_50.nii.gz
            mdt-model-fit ... --cl-device-ind 1
            mdt-model-fit ... --cl-device-ind {0, 1}
            mdt-model-fit ... --extra-protocol T1=T1_map.nii.gz T2=10
           ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('model', metavar='model', choices=mdt.get_models_list(),
                            help='model name, see mdt-list-models')
        parser.add_argument('dwi',
                            action=mdt.lib.shell_utils.get_argparse_extension_checker(['.nii', '.nii.gz', '.hdr', '.img']),
                            help='the diffusion weighted image').completer = FilesCompleter(['nii', 'gz', 'hdr', 'img'],
                                                                                            directories=False)
        parser.add_argument(
            'protocol', action=mdt.lib.shell_utils.get_argparse_extension_checker(['.prtcl']),
            help='the protocol file, see mdt-create-protocol').completer = FilesCompleter(['prtcl'],
                                                                                          directories=False)
        parser.add_argument('mask',
                            action=mdt.lib.shell_utils.get_argparse_extension_checker(['.nii', '.nii.gz', '.hdr', '.img']),
                            help='the (brain) mask to use').completer = FilesCompleter(['nii', 'gz', 'hdr', 'img'],
                                                                               directories=False)
        parser.add_argument('-o', '--output_folder',
                            help='the directory for the output, defaults to "output/<mask_name>" '
                                 'in the same directory as the dwi volume').completer = FilesCompleter()

        parser.add_argument('-n', '--noise-std', default=None,
                            help='the noise std, defaults to None for automatic noise estimation.'
                                 'Either set this to a value, or to a filename.')

        parser.add_argument('--gradient-deviations',
                            action=mdt.lib.shell_utils.get_argparse_extension_checker(['.nii', '.nii.gz', '.hdr', '.img']),
                            help="The volume with the gradient deviations to use, in HCP WUMINN format.").\
            completer = FilesCompleter(['nii', 'gz', 'hdr', 'img'], directories=False)

        parser.add_argument('--cl-device-ind', type=int, nargs='*', choices=self.available_devices,
                            help="The index of the device we would like to use. This follows the indices "
                                 "in mdt-list-devices and defaults to the first GPU.")

        parser.add_argument('--recalculate', dest='recalculate', action='store_true',
                            help="Recalculate the model(s) if the output exists. (default)")
        parser.add_argument('--no-recalculate', dest='recalculate', action='store_false',
                            help="Do not recalculate the model(s) if the output exists.")
        parser.set_defaults(recalculate=True)

        parser.add_argument('--only-recalculate-last', dest='only_recalculate_last', action='store_true',
                            help="Only recalculate the last model in a cascade. (default)")
        parser.add_argument('--recalculate-all', dest='only_recalculate_last', action='store_false',
                            help="Recalculate all models in a cascade.")
        parser.set_defaults(only_recalculate_last=True)

        parser.add_argument('--dont-use-cascaded-inits', dest='use_cascaded_inits', action='store_false',
                            help="Do not initialize the model with a better starting point.")
        parser.add_argument('--use-cascaded-inits', dest='use_cascaded_inits', action='store_true',
                            help="Initialize the model with a better starting point (default). "
                                 "Only works for default MDT models.")
        parser.set_defaults(use_cascaded_inits=True)

        parser.add_argument('--method', default='Powell',
                            choices=['Powell', 'Nelder-Mead', 'Levenberg-Marquardt', 'Subplex'],
                            help='The optimization method to use, defaults to Powell.')

        parser.add_argument('--patience', type=int, default=None,
                            help='The patience for the optimization routine')

        parser.add_argument('--double', dest='double_precision', action='store_true',
                            help="Calculate in double precision.")
        parser.add_argument('--float', dest='double_precision', action='store_false',
                            help="Calculate in single precision. (default)")
        parser.set_defaults(double_precision=False)

        parser.add_argument('--tmp-results-dir', dest='tmp_results_dir', default='True', type=str,
                            help='The directory for the temporary results. The default ("True") uses the config file '
                                 'setting. Set to the literal "None" to disable.').completer = FilesCompleter()

        parser.add_argument('--config-context', dest='config_context', type=str,
                            help='The configuration context to use during fitting the model. '
                                 'Same syntax as config files')

        parser.add_argument('--extra-protocol', dest='extra_protocol', type=str, nargs='+',
                            help='Additional protocol values, provide as <key>=<value> pairs')

        return parser

    def run(self, args, extra_args):
        mask_name = os.path.splitext(os.path.basename(os.path.realpath(args.mask)))[0]
        mask_name = mask_name.replace('.nii', '')
        output_folder = args.output_folder or os.path.join(os.path.dirname(args.dwi), 'output', mask_name)

        tmp_results_dir = args.tmp_results_dir
        for match, to_set in [('true', True), ('false', False), ('none', None)]:
            if tmp_results_dir.lower() == match:
                tmp_results_dir = to_set
                break

        noise_std = args.noise_std
        if noise_std is not None:
            if not os.path.isfile(os.path.realpath(noise_std)):
                noise_std = float(noise_std)

        def fit_model():
            input_data = mdt.load_input_data(
                os.path.realpath(args.dwi),
                os.path.realpath(args.protocol),
                os.path.realpath(args.mask),
                gradient_deviations=args.gradient_deviations,
                noise_std=noise_std,
                extra_protocol=get_extra_protocol(args.extra_protocol,
                                                  os.path.realpath('')))

            optimizer_options = {}
            if args.patience is not None:
                optimizer_options['patience'] = args.patience

            mdt.fit_model(args.model,
                          input_data,
                          output_folder,
                          method=args.method,
                          optimizer_options=optimizer_options,
                          recalculate=args.recalculate,
                          only_recalculate_last=args.only_recalculate_last,
                          cl_device_ind=args.cl_device_ind,
                          double_precision=args.double_precision,
                          tmp_results_dir=tmp_results_dir,
                          use_cascaded_inits=args.use_cascaded_inits)

        if args.config_context:
            with mdt.config_context(args.config_context):
                fit_model()
        else:
            fit_model()


def get_extra_protocol(extra_protocols_listing, base_dir):
    extra_protocol = {}

    if extra_protocols_listing:
        for argument in extra_protocols_listing:
            key, value = argument.split('=')

            if os.path.isfile(value):
                extra_protocol[key] = value
            elif os.path.isfile(base_dir + '/' + value):
                extra_protocol[key] = base_dir + '/' + value
            else:
                extra_protocol[key] = float(value)

    return extra_protocol


def get_doc_arg_parser():
    return ModelFit().get_documentation_arg_parser()


if __name__ == '__main__':
    ModelFit().start()
