import argparse
import os
import textwrap
import argcomplete
import sys

__author__ = 'Robbert Harms'
__date__ = "2015-10-16"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_argparse_extension_checker(choices, dir_allowed=False):
    """Get an :class:`argparge.Action` class that can check for correct extensions.

    Returns:
        argparse.Action: a class (not an instance) of an argparse action.
    """

    class Act(argparse.Action):
        def __call__(self, parser, namespace, fname, option_string=None):
            is_valid = any(map(lambda choice: fname[-len(choice):] == choice, choices))

            if not is_valid and dir_allowed and os.path.isdir(fname):
                is_valid = True

            if is_valid:
                setattr(namespace, self.dest, fname)
            else:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("File doesn't end with one of {}{}".format(choices, option_string))

    return Act


class BasicShellApplication(object):

    def __init__(self):
        self.parse_unknown_args = False

    @classmethod
    def console_script(cls):
        """Method used to start the command when launched from a distutils console script."""
        cls().start(sys.argv[1:])

    def start(self, run_args=None):
        """ Starts a command and registers single handlers.

        Args:
            run_args (:class:`list`): the list of run arguments. If None we use sys.argv[1:].
        """
        if run_args is None:
            run_args = sys.argv[1:]

        parser = self._get_arg_parser()
        argcomplete.autocomplete(parser)

        if self.parse_unknown_args:
            args, unknown = parser.parse_known_args(run_args)
            self.run(args, unknown)
        else:
            args = parser.parse_args(run_args)
            self.run(args, {})

    def run(self, args, extra_args):
        """Run the application with the given arguments.

        Args:
            extra_args:
            args: the arguments from the argparser.
        """

    def get_documentation_arg_parser(self):
        """Get the argument parser that can be used for writing the documentation

        Returns:
            argparse.ArgumentParser: the argument parser
        """
        return self._get_arg_parser(doc_parser=True)

    def _get_arg_parser(self, doc_parser=False):
        """Create the auto parser. This should be implemented by the implementing class.

        To enable autocomplete in your shell please execute activate-global-python-argcomplete in your shell.

        The arg parser should support two output modes, one for the command line (default) and one for generating
        the documentation. If the flag doc_parser is true the parser should be generated for the documentation.

        Args:
            doc_parser (boolean): If true the parser should be prepared for the documentation. If false (the default)
                the parser should be generated for the command line.

        Returns:
            argparse.ArgumentParser: the argument parser
        """
        description = textwrap.dedent("""
            Basic parser introduction here.

            Can be multiline.
        """)

        epilog = textwrap.dedent("""
            Examples of use:
                mdt-model-fit "BallStick (Cascade)" data.nii.gz data.prtcl roi_mask_0_50.nii.gz
        """)
        parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                         formatter_class=argparse.RawTextHelpFormatter)
        return parser

    def _format_examples(self, doc_parser, example_string):
        """Format the examples for either documentation or command line, with the given examples.

        This is commonly used in the epilog where you provide examples of use for your script.

        Args:
            doc_parser (boolean): if true we are preparing for the documentation, if false for a CLI
            example_string (str): the examples we wish to show
        """
        if doc_parser:
            return_str = 'Example_of_use::\n\n'
        else:
            return_str = 'Example_of_use:\n'

        for line in example_string.split('\n'):
            return_str += ' '*4 + line + '\n'

        return return_str
