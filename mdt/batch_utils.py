import glob
import os
import shutil
from six import string_types
from mdt.components_loader import BatchProfilesLoader
from mdt.protocols import load_from_protocol, load_bvec_bval
from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2015-08-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

# todo try to merge BatchSubjectInfo and BatchFitSubjectOutputInfo

class BatchSubjectInfo(object):

    def __init__(self, subject_id, dwi_fname, protocol, found_items):
        """This class contains all the information about found subjects during batch fitting.

        It is returned by the method get_subjects() from the class BatchProfile.

        Args:
            subject_id (str): the subject id
            dwi_fname (str): the filename with path to the dwi image
            protocol (Protocol): the protocol object to use
            found_items (dict): a dictionary with the found subject items.
                Can contain items like: 'dwi', 'bval', 'bvec', 'prtcl' and 'mask'
        """
        self.subject_id = subject_id
        self.dwi_fname = dwi_fname
        self.protocol = protocol
        self.found_items = found_items


class BatchFitSubjectOutputInfo(object):

    def __init__(self, path, subject_id, mask_name, model_name):
        """This class is used in conjunction with the function run_function_on_batch_fit_output().

        When applied the function run_function_on_batch_fit_output() calls the given function for every subject
        with as single argument an instance of this class. The user can then use this class to get information
        about the loaded subject.

        Args:
            output_path (str): the full path to the directory with the maps
            subject_id (str): the id of the subject
            mask_name (str): the name of the mask (not a path)
            model_name (str): the name of the model (not a path)
        """
        self.output_path = path
        self.subject_id = subject_id
        self.mask_name = mask_name
        self.model_name = model_name
        self.available_map_names = [split_image_path(v)[1] for v in glob.glob(os.path.join(self.output_path, '*.nii*'))]


class BatchProfile(object):

    def __init__(self, root_dir):
        """Instantiate this BatchProfile on the given root directory.

        Args:
            root_dir (str): the root directory from which to return all the subjects
        """
        self._root_dir = root_dir

    def get_root_dir(self):
        """Get the root dir this profile uses.

        Returns:
            str: the root dir this batch profile uses.
        """
        return self._root_dir

    def get_batch_fit_config_options(self):
        """Get the specific options from this batch fitting profile that will override the default config options.

        This should only return the content of the base config setting 'batch_fitting'.

        See the BatchFitting class for the resolution order of the config files.
        """

    def get_output_directory(self, subject_id):
        """Get the output directory for the subject with the given subject id.

        Args:
            subject_id (str): the subject id for which to get the output directory.
        """

    def get_subjects(self):
        """Get the information about all the subjects in the current folder.

        Returns:
            list of BatchSubjectInfo: the information about the found subjects
        """

    def profile_suitable(self):
        """Check if this directory can be used to load subjects from using this batch fitting profile.

        This is used for auto detecting the best batch fitting profile to use for loading
        subjects from the given root dir.

        Returns:
            boolean: true if this batch fitting profile can load datasets from this root directory, false otherwise.
        """

    def get_subjects_count(self):
        """Get the number of subjects this batch fitting profile can load from the current root directory.

        Returns:
            int: the number of subjects this batch fitting profile can load from the given directory.
        """


class SimpleBatchProfile(BatchProfile):

    def __init__(self, root_dir):
        super(SimpleBatchProfile, self).__init__(root_dir)
        self._subjects_found = self._get_subjects()

    def get_batch_fit_config_options(self):
        return {}

    def get_output_directory(self, subject_id):
        return os.path.join(self._root_dir, subject_id, 'output')

    def get_subjects(self):
        return self._subjects_found

    def profile_suitable(self):
        return len(self._subjects_found) > 0

    def get_subjects_count(self):
        return len(self._subjects_found)

    def _get_subjects(self):
        """Get the matching subjects from the given root dir.

        If a protocol file or brain mask can not be found for a subject this class will have to create one.

        You can use the routine _generate_brain_mask() and _create_protocol() for that, or overwrite them with
        your own implementation.

        Returns:
            list of BatchSubjectInfo: the information about the found subjects
        """
        return []

    def _get_protocol(self, found_items):
        """Get a protocol object that we can use during optimization.

        If no protocol file is present it will have to create one itself.
        This should not write a protocol file, it should only create and return a protocol object.

        Args:
            subject_id (str): the id of this subject
            found_items (dict): the found items during directory scanning

        Returns:
            protocol: a generated protocol object
        """
        if 'prtcl' in found_items:
            protocol = load_from_protocol(found_items['prtcl'])
        else:
            protocol = load_bvec_bval(found_items['bvec'], found_items['bval'])

        for col in ('TE', 'Delta', 'delta'):
            if col in found_items:
                protocol.add_column_from_file(col, found_items[col])

        return protocol

    def _get_basename(self, file):
        """Get the basename of a file. That is, the name of the files with all extensions stripped."""
        return split_image_path(file)[1]


class BatchFitOutputInfo(object):

    def __init__(self, data_folder, batch_profile_class=None):
        """Single point of information about batch fitting output.

        Args:
            data_folder (str): The data folder with the output files
            batch_profile_class (BatchProfile class or str): the batch profile class to use, can also be the name
                of a batch profile to load. If not given it is auto detected.
                Please note it expects a callable that returns a batch profile instance. For example, you can use it as:
                    batch_profile_class=MyBatchProfile
                but this would not work:
                    batch_profile_class=MyBatchProfile()
        """
        self._data_folder = data_folder
        self._batch_profile = batch_profile_factory(batch_profile_class, data_folder)
        self._subjects = self._batch_profile.get_subjects()
        self._subjects_dirs = {subject_info.subject_id:
                                   self._batch_profile.get_output_directory(subject_info.subject_id)
                               for subject_info in self._subjects}
        self._mask_paths = {}

    def get_available_masks(self):
        """Searches all the subjects and lists the unique available masks.

        Returns:
            list: the list of the available maps. Not all subjects may have the available mask.
        """
        s = set()
        for subject_id, path in self._subjects_dirs.items():
            masks = (p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p)))
            map(s.add, masks)
        return list(sorted(list(s)))

    def get_path_to_mask_per_subject(self, mask_name, error_on_missing_mask=False):
        """Get for every subject the path to the given mask name.

        If a subject does not have that mask_name it is either skipped or an error is raised, depending on the setting
        error_on_missing_mask.

        Args:
            mask_name (str): the name of the mask we return the path to per subject
            error_on_missing_mask (boolean): if we don't have the mask for one subject should we raise an error or skip
                the subject?

        Returns:
            dict: per subject ID the path to the mask
        """
        if mask_name in self._mask_paths:
            return self._mask_paths[mask_name]

        paths = {}
        for subject_id, path in self._subjects_dirs.items():
            mask_dir = os.path.join(path, mask_name)
            if os.path.isdir(mask_dir):
                paths.update({subject_id: mask_dir})
            else:
                if error_on_missing_mask:
                    raise ValueError('Missing the choosen mask "{0}" for subject "{1} '
                                     'and error_on_missing_mask is True"'.format(mask_name, subject_id))

        self._mask_paths.update({mask_name: paths})
        return paths

    def subject_output_info_generator(self, mask_name, error_on_missing_mask=False):
        """Generates for every subject an output info object which contains all relevant information about the subject.

        If a subject does not have that mask_name it is either skipped or an error is raised, depending on the setting
        error_on_missing_mask.

        Args:
            mask_name (str): the name of the mask we return the path to per subject
            error_on_missing_mask (boolean): if true, if we don't have the given mask for a subject we raise an error.
                If false, if we don't have a mask for a subject we do nothing.

        Returns:
            generator: returns an BatchFitSubjectOutputInfo per subject
        """
        mask_paths = self.get_path_to_mask_per_subject(mask_name, error_on_missing_mask)
        for subject_id, mask_path in mask_paths.items():
            for model_name in os.listdir(mask_path):
                output_path = os.path.join(mask_path, model_name)
                if os.path.isdir(output_path):
                    yield BatchFitSubjectOutputInfo(output_path, subject_id, mask_name, model_name)


def run_function_on_batch_fit_output(data_folder, func, batch_profile_class=None):
    """Run a function on the output of a batch fitting routine.

    This enables you to run a function on every model output from every subject. The python function should accept
    as single argument an instance of the class BatchFitSubjectOutputInfo.

    Args:
        data_folder (str): The data folder with the output files
        func (python function): the python function we should call for every map and model
        batch_profile_class (BatchProfile class or str): the batch profile class to use, can also be the name
            of a batch profile to load. If not given it is auto detected.
            Please note it expects a callable that returns a batch profile instance. For example, you can use it as:
                batch_profile_class=MyBatchProfile
            but this would not work:
                batch_profile_class=MyBatchProfile()
    """
    output_info = BatchFitOutputInfo(data_folder, batch_profile_class)
    mask_names = output_info.get_available_masks()
    for mask_name in mask_names:
        map(func, output_info.subject_output_info_generator(mask_name))


def batch_profile_factory(batch_profile_class, data_folder):
    """Wrapper function for getting a batch profile.

    Args:
        batch_profile_class (None, string or BatchProfile class): indication of the batch profile to load.
            If a BatchProfile class is given it expects a callable that can generate a batch profile instance.
            If a string is given it is loaded from the users home folder. Else the best matching profile is returned.
        data_folder (str): the data folder we want to use the batch profile on.

    Returns:
        If the given batch profile is None we return the output from get_best_batch_profile(). If batch profile is
        a string we load it from the batch profiles loader. Else we return the input.
    """
    if batch_profile_class is None:
        return get_best_batch_profile(data_folder)
    elif isinstance(batch_profile_class, string_types):
        return BatchProfilesLoader().load(batch_profile_class, data_folder)
    return batch_profile_class(data_folder)


def get_best_batch_profile(data_folder):
    """Get the batch profile that best matches the given directory.

    Args:
        directory (str): the directory for which to get the best batch profile.

    Returns:
        BatchProfile: the best matching batch profile.
    """
    profile_loader = BatchProfilesLoader()
    crawlers = [profile_loader.load(c, data_folder) for c in profile_loader.list_all()]

    best_crawler = None
    best_subjects_count = 0
    for crawler in crawlers:
        if crawler.profile_suitable():
            tmp_count = crawler.get_subjects_count()
            if tmp_count > best_subjects_count:
                best_crawler = crawler
                best_subjects_count = tmp_count

    return best_crawler


def collect_batch_fit_output(data_folder, output_dir, batch_profile_class=None, mask_name=None, symlink=False):
    """Load from the given data folder all the output files and put them into the output directory.

    If there is more than one mask file available the user has to choose which mask to use using the mask_name
    keyword argument. If it is not given an error is raised.

    The results for the chosen mask it placed in the output folder per subject. Example:
        <output_dir>/<subject_id>/<results>

    Args:
        data_folder (str): The data folder with the output files
        output_dir (str): The path to the output folder where all the files will be put.
        batch_profile_class (BatchProfile class or str): the batch profile class to use, can also be the name
            of a batch profile to load. If not given it is auto detected.
            Please note it expects a callable that returns a batch profile instance. For example, you can use it as:
                batch_profile_class=MyBatchProfile
            but this would not work:
                batch_profile_class=MyBatchProfile()
        mask_name (str): the mask to use to get the output from
        symlink (boolean): only available under Unix OS's. Creates a symlink instead of copying.
    """
    output_info = BatchFitOutputInfo(data_folder, batch_profile_class)
    mask_names = output_info.get_available_masks()
    if len(mask_names) > 1:
        if mask_name is None:
            raise ValueError('There are results of more than one mask. '
                             'Please choose one out of ({}) '
                             'using the \'mask_name\' keyword.'.format(', '.join(mask_names)))
    else:
        mask_name = mask_names[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask_paths = output_info.get_path_to_mask_per_subject(mask_name)

    for subject_id, mask_path in mask_paths.items():
        subject_out = os.path.join(output_dir, subject_id)

        if os.path.exists(subject_out):
            if os.path.islink(subject_out):
                os.unlink(subject_out)
            else:
                shutil.rmtree(output_dir)

        if symlink:
            os.symlink(mask_path, subject_out)
        else:
            shutil.copytree(mask_path, subject_out)