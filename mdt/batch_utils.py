import glob
import logging
import os
import shutil
from six import string_types
from mdt.components_loader import BatchProfilesLoader
from mdt.data_loaders.protocol import ProtocolLoader
from mdt.masking import create_write_median_otsu_brain_mask
from mdt.protocols import load_protocol, load_bvec_bval
from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2015-08-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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

    def get_subjects(self):
        """Get the information about all the subjects in the current folder.

        Returns:
            list of SubjectInfo: the information about the found subjects
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
        """A base class for quickly implementing a batch profile.

        Implementing classes need only implement the method _get_subjects(). This class will handle the rest.
        """
        super(SimpleBatchProfile, self).__init__(root_dir)
        self._subjects_found = self._get_subjects()

    def get_batch_fit_config_options(self):
        return {}

    def get_subjects(self):
        return self._subjects_found

    def profile_suitable(self):
        return len(self._subjects_found) > 0

    def get_subjects_count(self):
        return len(self._subjects_found)

    def _get_subjects(self):
        """Get the matching subjects from the given root dir.

        This is the only function that should be implemented by implementing classes to get up and running.

        Returns:
            list of SubjectInfo: the information about the found subjects
        """
        return []


class SubjectInfo(object):

    @property
    def subject_id(self):
        """Get the ID of this subject.

        Returns:
            str: the id of this subject
        """
        return ''

    @property
    def output_dir(self):
        """Get the output folder for this subject.

        Returns:
            str: the output folder
        """
        return ''

    def get_protocol_loader(self):
        """Get the protocol to use, or a filename of a protocol file to load.

        Returns:
            ProtocolLoader: the protocol loader
        """

    def get_dwi_info(self):
        """Get the diffusion weighted image information.

        Returns:
            (img, header) tuple or str: either a string with the filename of the image to load or the actual
                image itself with a header in a tuple.
        """

    def get_mask_filename(self):
        """Get the filename of the mask to load.

        Returns:
            str: the filename of the mask to load
        """

    def get_gradient_deviations(self):
        """Get a possible gradient deviation image to use.

        Returns:
            str: the filename of the gradient deviations to use, None if not applicable.
        """
        return None


class SimpleSubjectInfo(SubjectInfo):

    def __init__(self, subject_id, dwi_fname, protocol_loader, mask_fname, output_dir, gradient_deviations=None):
        """This class contains all the information about found subjects during batch fitting.

        It is returned by the method get_subjects() from the class BatchProfile.

        Args:
            subject_id (str): the subject id
            dwi_fname (str): the filename with path to the dwi image
            protocol_loader (ProtocolLoader): the protocol loader that can load us the protocol
            mask_fname (str): the filename of the mask to load. If None a mask is auto generated.
            output_dir (str): the
        """
        self._subject_id = subject_id
        self._dwi_fname = dwi_fname
        self._protocol_loader = protocol_loader
        self._mask_fname = mask_fname
        self._output_dir = output_dir
        self._gradient_deviations = gradient_deviations

        if self._mask_fname is None:
            self._mask_fname = os.path.join(self.output_dir, 'auto_generated_mask.nii.gz')

    @property
    def subject_id(self):
        return self._subject_id

    @property
    def output_dir(self):
        return self._output_dir

    def get_subject_id(self):
        return self.subject_id

    def get_protocol_loader(self):
        return self._protocol_loader

    def get_dwi_info(self):
        return self._dwi_fname

    def get_mask_filename(self):
        if not os.path.isfile(self._mask_fname):
            logger = logging.getLogger(__name__)
            logger.info('Creating a brain mask for subject {0}'.format(self.subject_id))

            protocol = self.get_protocol_loader().get_protocol()
            create_write_median_otsu_brain_mask(self.get_dwi_info(), protocol, self._mask_fname)

        return self._mask_fname

    def get_gradient_deviations(self):
        return self._gradient_deviations


class BatchFitProtocolLoader(ProtocolLoader):

    def __init__(self, prtcl_fname=None, bvec_fname=None, bval_fname=None,
                 extra_cols=None, extra_cols_from_file=None):
        """A simple protocol loader for loading a protocol from a protocol file or bvec/bval files.

        The parameters extra_cols and extra_cols_from_file allow you to add additional columns
        to the loaded protocol.

        Please note that extra_cols_from_file is loaded first and extra_cols loaded second.
        This means that the values in extra_cols overwrite the other columns if present.

        Args:
            prtcl_fname (str): the filename of the protocol file to (preferably load)
            bvec_fname (str): the filename of the bvec file to use if no protocol file is given
            bval_fname (str): the filename of the bval file to use if no protocol file is given
            extra_cols (dict): a dictionary with additional columns to add. The keys are column names
                and the values as values of that column. Can be one value or one per row of the protocol.
                Example: {'TE': 0.01, 'Delta': 0.01, ...}
            extra_cols_from_file (dict): Load additional columns from file. The keys are column names and
                the values the values of that column.
                Example: {'TE': '/path/to/TE_file', 'TI': '/path/to/T1_file', ...}
        """
        super(BatchFitProtocolLoader, self).__init__()
        self._prtcl_fname = prtcl_fname
        self._bvec_fname = bvec_fname
        self._bval_fname = bval_fname
        self._extra_cols = extra_cols
        self._extra_cols_from_file = extra_cols_from_file

    def get_protocol(self):
        super(BatchFitProtocolLoader, self).get_protocol()
        if self._prtcl_fname and os.path.isfile(self._prtcl_fname):
            protocol = load_protocol(self._prtcl_fname)
        else:
            protocol = load_bvec_bval(self._bvec_fname, self._bval_fname)

        if self._extra_cols_from_file:
            for name, filename in self._extra_cols_from_file.items():
                protocol.add_column_from_file(name, filename)

        if self._extra_cols:
            for name, value in self._extra_cols.items():
                protocol.add_column(name, value)

        return protocol


class BatchFitSubjectOutputInfo(object):

    def __init__(self, subject_info, output_path, mask_name, model_name):
        """This class is used in conjunction with the function run_function_on_batch_fit_output().

        Args:
            subject_info (SubjectInfo): the information about the subject before batch fitting
            output_path (str): the full path to the directory with the maps
            mask_name (str): the name of the mask (not a path)
            model_name (str): the name of the model (not a path)
        """
        self.subject_info = subject_info
        self.output_path = output_path
        self.mask_name = mask_name
        self.model_name = model_name
        self.available_map_names = [split_image_path(v)[1] for v in glob.glob(os.path.join(self.output_path, '*.nii*'))]


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
        self._subjects_dirs = {subject_info.subject_id: subject_info.output_dir for subject_info in self._subjects}
        self._mask_paths = {}

    def get_available_masks(self):
        """Searches all the subjects and lists the unique available masks.

        Returns:
            list: the list of the available maps. Not all subjects may have the available mask.
        """
        s = set()
        for subject_id, path in self._subjects_dirs.items():
            masks = (p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p)))
            list(map(s.add, masks))
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

        for subject_info in self._subjects:
            if subject_info.subject_id in mask_paths:
                mask_path = mask_paths[subject_info.subject_id]

                for model_name in os.listdir(mask_path):
                    output_path = os.path.join(mask_path, model_name)
                    if os.path.isdir(output_path):
                        yield BatchFitSubjectOutputInfo(subject_info, output_path, mask_name, model_name)
            else:
                if error_on_missing_mask:
                    raise ValueError('Missing the choosen mask "{0}" for subject "{1} '
                                     'and error_on_missing_mask is True"'.format(mask_name, subject_info.subject_id))


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
        list(map(func, output_info.subject_output_info_generator(mask_name)))


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

        if os.path.exists(subject_out) or os.path.islink(subject_out):
            if os.path.islink(subject_out):
                os.unlink(subject_out)
            else:
                shutil.rmtree(output_dir)

        if symlink:
            os.symlink(mask_path, subject_out)
        else:
            shutil.copytree(mask_path, subject_out)