import datetime
import hashlib
import os

from six import string_types

__author__ = 'Robbert Harms'
__date__ = "2016-08-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SaveUserScriptInfo(object):

    def __init__(self):
        """Base class for writing the user info to a file.
        """

    def write(self, output_file):
        """Write the information about the script the user is executing to a output information file.

        Args:
            output_file (str): the file we will write the output to
        """
        raise NotImplementedError


def easy_save_user_script_info(save_user_script_info, output_file, file_path_from_stack):
    """Handy routine for saving the user script info from multiple sources.

    Args:
        save_user_script_info (boolean, str or SaveUserScriptInfo): The info we need to save about the script the
            user is currently executing. If True (default) we use the stack to lookup the script the user is executing
            and save that using a SaveFromScript saver. If a string is given we use that filename again for the
            SaveFromScript saver. If False or None, we do not write any information. If a SaveUserScriptInfo is
            given we use that directly.
        output_file (str): the output folder for the output file
        file_path_from_stack (str): the file path from the stack inspection
    """
    if save_user_script_info:
        if isinstance(save_user_script_info, SaveUserScriptInfo):
            save_user_script_info.write(output_file)

        elif isinstance(save_user_script_info, string_types):
            SaveFromScript(save_user_script_info).write(output_file)

        elif save_user_script_info is True:
            SaveFromScript(file_path_from_stack).write(output_file)


class SaveFromScript(SaveUserScriptInfo):

    def __init__(self, user_script_path):
        super(SaveFromScript, self).__init__()
        self._user_script_path = user_script_path

    def write(self, output_file):
        """Write the information about the script the user is executing to a output information file.

        This function relies on the caller of the script to provide the filename of the script he user is currently
        executing. You can do this using for example a stack lookup:

        .. code-block:: python

            stack()[1][0].f_globals.get('__file__')

        Args:
            output_file (str): the file we will write the output to
        """
        output_file_existed = os.path.exists(output_file)

        utc_time = datetime.datetime.utcnow()

        class_hash_name = hashlib.sha1()
        class_hash_name.update(str(utc_time).encode('utf-8'))

        with open(output_file, 'a') as output_file:
            if not output_file_existed:
                output_file.write('from mdt.user_script_info import UserScriptInfo')

            output_file.write('\n\n\nclass Script_{}(UserScriptInfo):\n\n'.format(class_hash_name.hexdigest()))
            output_file.write(' '*4 + 'date = "{}"\n'.format(utc_time))
            output_file.write(' '*4 + 'filename = "{}"\n\n'.format(self._user_script_path))
            output_file.write(' ' * 4 + '@staticmethod\n'.format(self._user_script_path))
            output_file.write(' ' * 4 + 'def body():\n'.format(self._user_script_path))

            with open(self._user_script_path, 'r') as input_file:
                for line in input_file:
                    output_file.write(' '*8 + line)


class UserScriptInfo(object):
    pass
