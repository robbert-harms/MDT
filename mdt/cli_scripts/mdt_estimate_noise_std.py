#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Estimate the noise standard deviation of the Gaussian noise in the original complex image domain.

The result is a single floating point number with the noise std. for every voxel. It uses the estimation routines
from the components folders for the estimation. The estimation is the same as the one used in mdt-model-fit, but
since the noise std estimation depends on the mask used, it is better to call this function beforehand with a
complete brain mask. Later, the mdt-model-fit routine can be called on smaller masks with as noise std the value
from this function.
"""
import argparse
import os
import mdt
from argcomplete.completers import FilesCompleter
from mdt.lib.shell_utils import BasicShellApplication
from mot.lib import cl_environments
import textwrap


__author__ = 'Robbert Harms'
__date__ = "2015-08-18"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NoiseStdEstimation(BasicShellApplication):

    def __init__(self):
        super().__init__()
        self.available_devices = list((ind for ind, env in
                                       enumerate(cl_environments.CLEnvironmentFactory.smart_device_selection())))

    def _get_arg_parser(self, doc_parser=False):
        description = textwrap.dedent(__doc__)

        examples = textwrap.dedent('''
            mdt-estimate-noise-std data.nii.gz data.prtcl full_mask.nii.gz
        ''')
        epilog = self._format_examples(doc_parser, examples)

        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)
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

        return parser

    def run(self, args, extra_args):
        input_data = mdt.load_input_data(os.path.realpath(args.dwi),
                                         os.path.realpath(args.protocol),
                                         os.path.realpath(args.mask))

        with mdt.with_logging_to_debug():
            noise_std = mdt.estimate_noise_std(input_data)
            print(noise_std)


def get_doc_arg_parser():
    return NoiseStdEstimation().get_documentation_arg_parser()


if __name__ == '__main__':
    NoiseStdEstimation().start()
