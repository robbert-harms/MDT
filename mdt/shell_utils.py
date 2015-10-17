import argparse
import textwrap

__author__ = 'Robbert Harms'
__date__ = "2015-10-16"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_argparse_extension_checker(choices):

    class Act(argparse.Action):
        def __call__(self, parser, namespace, fname, option_string=None):
            is_valid = any(map(lambda choice: fname[-len(choice):] == choice, choices))
            if is_valid:
                setattr(namespace, self.dest, fname)
            else:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("File doesn't end with one of {}{}".format(choices, option_string))

    return Act


def get_citation_message():
    """The citation message used in the shell scripts.

    Returns:
        str: the citation message for use in the description of every shell script
    """
    return textwrap.dedent("""
        If you use any of the scripts/functions/tools from MDT in your research, please cite the following paper:
            <citation here>
    """)