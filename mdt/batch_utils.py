"""Routines for fitting models on multiple subjects.

The most important part of this are the batch profiles. These encapsulate information about the subjects
and about the modelling settings. Suppose you have a directory full of subjects that you want to analyze with a
few models. One way to do that is to write some scripts yourself that walk through the directory and fit
the models to the subjects. The other way would be to implement a :class:`BatchProfile` that contains details
about your directory structure and let :func:`mdt.batch_fit` fetch all the subjects for you.

Batch profiles contain a list with subject information (see :class:`SubjectInfo`) and a list of models
we wish to apply to these subjects. Furthermore each profile should support some functionality that checks
if this profile is suitable for a given directory. Using those functions the :func:`mdt.batch_fit` can try
to auto-recognize the batch profile to use based on the profile that is suitable and returns the most subjects.
"""
import glob
import logging
import os
import shutil
from textwrap import dedent

import six
from six import string_types
from mdt.components_loader import BatchProfilesLoader
from mdt.data_loaders.protocol import ProtocolLoader
from mdt.masking import create_median_otsu_brain_mask
from mdt.protocols import load_protocol, auto_load_protocol
from mdt.utils import split_image_path, AutoDict, load_input_data
from mdt.nifti import load_nifti

__author__ = 'Robbert Harms'
__date__ = "2015-08-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BatchProfile(object):

    def get_subjects(self, data_folder):
        """Get the information about all the subjects in the current folder.

        Args:
            data_folder (str): the data folder from which to load the subjects

        Returns:
            list of :class:`SubjectInfo`: the information about the found subjects
        """
        raise NotImplementedError()

    def is_suitable(self, data_folder):
        """Check if this directory can be used to use subjects from using this batch fitting profile.

        This is used for auto detecting the best batch fitting profile to use for loading
        subjects from the given base dir.

        Args:
            data_folder (str): the data folder from which to load the subjects

        Returns:
            boolean: true if this batch fitting profile can use the subjects in the current base directory,
                false otherwise.
        """
        raise NotImplementedError()

    def get_subjects_count(self, data_folder):
        """Get the number of subjects this batch fitting profile can use from the current base directory.

        Args:
            data_folder (str): the data folder from which to load the subjects

        Returns:
            int: the number of subjects this batch fitting profile can use from the current base directory.
        """
        raise NotImplementedError()


class SimpleBatchProfile(BatchProfile):

    def __init__(self, *args, **kwargs):
        """A base class for quickly implementing a batch profile.

        Implementing classes need only implement the method :meth:`_get_subjects`, then this class will handle the rest.

        Args:
            base_directory (str): the base directory from which we will load the subjects information
        """
        super(SimpleBatchProfile, self).__init__()

    def get_subjects(self, data_folder):
        return self._get_subjects(data_folder)

    def is_suitable(self, data_folder):
        return self.get_subjects_count(data_folder) > 0

    def get_subjects_count(self, data_folder):
        return len(self.get_subjects(data_folder))

    def _autoload_noise_std(self, data_folder, subject_id, file_path=None):
        """Try to autoload the noise standard deviation from a noise_std file.

        Args:
            data_folder (str): the data base folder
            subject_id (str): the subject for which to use the noise std.
            file_path (str): optionally provide the exact file to use.

        Returns:
            float or None: a float if a float could be loaded from a file noise_std, else nothing.
        """
        file_path = file_path or os.path.join(data_folder, subject_id, 'noise_std')
        noise_std_files = glob.glob(file_path + '*')
        if len(noise_std_files):
            with open(noise_std_files[0], 'r') as f:
                return float(f.read())
        return None

    def _get_subjects(self, data_folder):
        """Get the matching subjects from the given base dir.

        This is the only function that should be implemented to get up and running.

        Args:
            data_folder (str): the data folder from which to load the subjects

        Returns:
            list of SubjectInfo: the information about the found subjects
        """
        return []

    def _get_first_existing_file(self, filenames, default=None, prepend_path=None):
        """Tries a list of filenames and returns the first filename in the list that exists.

        Args:
            filenames (iterator): the list of filenames to search for existence
            default (str): the default value returned if none of the filenames existed
            prepend_path (str): the path to optionally prepend to every file before checking existence

        Returns:
            str: the filename of the first existing file, if prepend path is set it is included.
        """
        for fname in filenames:
            if fname:
                if prepend_path:
                    fname = os.path.join(prepend_path, fname)
                if os.path.isfile(fname):
                    return fname
        return default

    def _get_first_existing_nifti(self, filenames, default=None, prepend_path=None):
        """Tries a list of filenames and returns the first filename in the list that exists.

        Additional to the method :meth:`_get_first_existing_file`, this additionally tries to see for every filename
        if a file with extension '.nii' or with '.nii.gz' exists (in that order) for that filename. If so,
        the path with the added extension is returned.

        Args:
            filenames (iterator): the list of filenames to search for existence, does additional extension lookup
                per filename
            default (str): the default value returned if none of the filenames existed
            prepend_path (str): the path to optionally prepend to every file before checking existence

        Returns:
            str: the filename of the first existing file, can contain an extra extension for the returned filename.
        """
        for fname in filenames:
            resolve_extension = self._get_first_existing_file([fname, fname + '.nii', fname + '.nii.gz'],
                                                              prepend_path=prepend_path)
            if resolve_extension:
                return resolve_extension
        return default

    def _autoload_protocol(self, path, protocols_to_try=(), bvecs_to_try=(), bvals_to_try=(), protocol_columns=None):
        prtcl_fname = self._get_first_existing_file(protocols_to_try, prepend_path=path)
        bval_fname = self._get_first_existing_file(bvals_to_try, prepend_path=path)
        bvec_fname = self._get_first_existing_file(bvecs_to_try, prepend_path=path)

        return BatchFitProtocolLoader(
            path, protocol_fname=prtcl_fname, bvec_fname=bvec_fname,
            bval_fname=bval_fname, protocol_columns=protocol_columns)


class SubjectInfo(object):

    @property
    def subject_id(self):
        """Get the ID of this subject.

        Returns:
            str: the id of this subject
        """
        raise NotImplementedError()

    @property
    def subject_base_folder(self):
        """Get the data base folder of this subject.

        Returns:
            str: the folder with the main data of this subject, this subject's home folder.
        """
        raise NotImplementedError()

    def get_input_data(self, use_gradient_deviations=False):
        """Get the input data for this subject.

        This is the data we will use during model fitting.

        Args:
            use_gradient_deviations (boolean): if we should enable the use of the gradient deviations

        Returns:
            :class:`~mdt.utils.MRIInputData`: the input data to use during model fitting
        """
        raise NotImplementedError()


class SimpleSubjectInfo(SubjectInfo):

    def __init__(self, subject_base_folder, subject_id, dwi_fname, protocol_loader,
                 mask_fname, gradient_deviations=None, noise_std=None):
        """This class contains all the information about found subjects during batch fitting.

        It is returned by the method get_subjects() from the class BatchProfile.

        Args:
            subject_base_folder (str): the base folder of this subject
            subject_id (str): the subject id
            dwi_fname (str): the filename with path to the dwi image
            protocol_loader (ProtocolLoader): the protocol loader that can use us the protocol
            mask_fname (str): the filename of the mask to use. If None a mask is auto generated.
            gradient_deviations (str) if given, the path to the gradient deviations
            noise_std (float, ndarray, str): either None for automatic noise detection or a float with the noise STD
                to use during fitting or an ndarray with one value per voxel.
        """
        self._subject_base_folder = subject_base_folder
        self._subject_id = subject_id
        self._dwi_fname = dwi_fname
        self._protocol_loader = protocol_loader
        self._mask_fname = mask_fname
        self._gradient_deviations = gradient_deviations
        self._noise_std = noise_std

    @property
    def subject_id(self):
        return self._subject_id

    @property
    def subject_base_folder(self):
        return self._subject_base_folder

    def get_input_data(self, use_gradient_deviations=False):
        if use_gradient_deviations:
            gradient_deviations = self._get_gradient_deviations()
        else:
            gradient_deviations = None

        return load_input_data(self._dwi_fname,
                               self._protocol_loader.get_protocol(),
                               self._get_mask(),
                               gradient_deviations=gradient_deviations,
                               noise_std=self._noise_std)

    def _get_mask(self):
        if self._mask_fname is None or not os.path.isfile(self._mask_fname):
            logger = logging.getLogger(__name__)
            logger.info('Creating a brain mask for subject {0}'.format(self.subject_id))

            protocol = self._protocol_loader.get_protocol()
            return create_median_otsu_brain_mask(self._dwi_fname, protocol)
        return self._mask_fname

    def _get_gradient_deviations(self):
        if self._gradient_deviations is not None:
            return load_nifti(self._gradient_deviations).get_data()
        return None


class BatchSubjectSelection(object):

    def get_selection(self, subject_ids):
        """Get the selection of subjects from the given list of subjects.

        Args:
            subject_ids (list of str): the list of subject ids from which we can choose which one to process

        Returns:
            list of str: the subject ids we want to use
        """
        raise NotImplementedError()


class AllSubjects(BatchSubjectSelection):

    def __init__(self):
        """Selects all subjects for use in the processing"""
        super(AllSubjects, self).__init__()

    def get_selection(self, subject_ids):
        return subject_ids


class SelectedSubjects(BatchSubjectSelection):

    def __init__(self, subject_ids=None, indices=None, start_from=None):
        """Only process the selected subjects.

        This method allows either a selection by index (unsafe for the order may change) or by subject name/ID (more
        safe in general). If ``start_from`` is given it additionally limits the list of selected subjects to include
        only those after that index.

        This essentially creates three different subsets of the given list of subjects and it will only process
        the subjects in the intersection of all those sets.

        Set any one of the options to None to ignore that option.

        Args:
            subject_ids (list of str): the list of names of subjects to process
            indices (list/tuple of int): the list of indices of subjects we wish to process
            start_from (list or int): the index of the name of the subject from which we want to start processing.
        """
        self.subject_ids = subject_ids
        self.indices = indices
        self.start_from = start_from

    def get_selection(self, subject_ids):
        starting_pos = self._get_starting_pos(subject_ids)

        if self.indices is None and self.subject_ids is None:
            return subject_ids[starting_pos:]

        return_ids_ind = []
        if self.indices:
            return_ids_ind = [subject for ind, subject in enumerate(subject_ids)
                              if ind in self.indices and ind >= starting_pos]

        return_ids_id = []
        if self.subject_ids:
            return_ids_id = list(filter(lambda subject_id: subject_id in self.subject_ids, subject_ids))

        return list(set(return_ids_id + return_ids_ind))

    def _get_starting_pos(self, subject_ids):
        if self.start_from is None:
            return 0

        if isinstance(self.start_from, six.string_types):
            for ind, subject_id in enumerate(subject_ids):
                if subject_id == self.start_from:
                    return ind

        for ind, subject_id in enumerate(subject_ids):
            if ind == int(self.start_from):
                return ind


class BatchFitProtocolLoader(ProtocolLoader):

    def __init__(self, base_dir, protocol_fname=None, protocol_columns=None, bvec_fname=None, bval_fname=None):
        """A simple protocol loader for loading a protocol from a protocol file or bvec/bval files.

        This either loads the protocol file if present, or autoloads the protocol using the auto_load_protocol
        from the protocol module.
        """
        super(BatchFitProtocolLoader, self).__init__()
        self._base_dir = base_dir
        self._protocol_fname = protocol_fname
        self._bvec_fname = bvec_fname
        self._bval_fname = bval_fname
        self._protocol_columns = protocol_columns

    def get_protocol(self):
        super(BatchFitProtocolLoader, self).get_protocol()

        if self._protocol_fname and os.path.isfile(self._protocol_fname):
            return load_protocol(self._protocol_fname)

        return auto_load_protocol(self._base_dir, protocol_columns=self._protocol_columns,
                                  bvec_fname=self._bvec_fname, bval_fname=self._bval_fname)


def batch_profile_factory(batch_profile, base_directory):
    """Wrapper function for getting a batch profile.

    Args:
        batch_profile (None, string or BatchProfile): indication of the batch profile to use.
            If a string is given it is loaded from the users home folder. Else the best matching profile is returned.
        base_directory (str): the data folder we want to use the batch profile on.

    Returns:
        BatchProfile: If the given batch profile is None we return the output from get_best_batch_profile().
            If batch profile is a string we use it from the batch profiles loader. Else we return the input.
    """
    if batch_profile is None:
        return get_best_batch_profile(base_directory)
    elif isinstance(batch_profile, string_types):
        return BatchProfilesLoader().load(batch_profile)
    return batch_profile


def get_best_batch_profile(data_folder):
    """Get the batch profile that best matches the given directory.

    Args:
        data_folder (str): the directory for which to get the best batch profile.

    Returns:
        BatchProfile: the best matching batch profile.
    """
    profile_loader = BatchProfilesLoader()
    profiles = [profile_loader.load(profile) for profile in profile_loader.list_all()]

    best_crawler = None
    best_subjects_count = 0
    for profile in profiles:
        if profile.is_suitable(data_folder):
            tmp_count = profile.get_subjects_count(data_folder)
            if tmp_count > best_subjects_count:
                best_crawler = profile
                best_subjects_count = tmp_count

    return best_crawler


def batch_apply(func, data_folder, batch_profile=None, subjects_selection=None, extra_args=None):
    """Apply a function on the subjects found in the batch profile.

    Args:
        func (callable): the function we will apply for every subject, should accept as single argument an instance of
            :class:`SubjectInfo`.
        data_folder (str): The data folder to process
        batch_profile (:class:`~mdt.batch_utils.BatchProfile` or str): the batch profile to use,
            or the name of a batch profile to use. If not given it is auto detected.
        subjects_selection (:class:`~mdt.batch_utils.BatchSubjectSelection`): the subjects to use for processing.
            If None all subjects are processed.
        extra_args (list): a list of additional arguments that are passed to the function. If this is set,
            the callback function must accept these additional args.

    Returns:
        dict: per subject id the output from the function
    """
    batch_profile = batch_profile_factory(batch_profile, data_folder)
    subjects_selection = subjects_selection or AllSubjects()

    if batch_profile is None:
        raise RuntimeError('No suitable batch profile could be '
                           'found for the directory {0}'.format(os.path.abspath(data_folder)))

    selected_subjects = subjects_selection.get_selection(
        [el.subject_id for el in batch_profile.get_subjects(data_folder)])
    subjects = [subject for subject in batch_profile.get_subjects(data_folder)
                if subject.subject_id in selected_subjects]

    results = {}
    for subject in subjects:
        def f(subject):
            if extra_args:
                return func(subject, *extra_args)
            return func(subject)

        results[subject.subject_id] = f(subject)
    return results


class BatchFitSubjectOutputInfo(object):

    def __init__(self, output_path, subject_id, model_name):
        """This class is used in conjunction with the function :func:`run_function_on_batch_fit_output`.

        Args:
            output_path (str): the full path to the directory with the maps
            subject_id (str): the id of the current subject
            model_name (str): the name of the model (not a path)
        """
        self.subject_id = subject_id
        self.output_path = output_path
        self.model_name = model_name

    def __repr__(self):
        return dedent('''
            subject_id: {subject_id}
            output_path: {output_path}
            model_name: {model_name}
        '''.format(subject_id=self.subject_id, output_path=self.output_path, model_name=self.model_name)).strip()


def run_function_on_batch_fit_output(func, output_folder, subjects_selection=None):
    """Run a function on the output of a batch fitting routine.

    This enables you to run a function on every model output from every subject. This expects the output directory
    to contain directories and files like <subject_id>/<model_name>/<map_name>.nii.gz

    Args:
        func (python function): the python function we should call for every map and model.
            This should accept as single parameter a :class:`BatchFitSubjectOutputInfo`.
        output_folder (str): The data input folder
        subjects_selection (BatchSubjectSelection): the subjects to use for processing.
            If None all subjects are processed.

    Returns:
        dict: indexed by subject->model_name, values are the return values of the users function
    """
    subject_ids = list(os.listdir(output_folder))
    if subjects_selection:
        subject_ids = subjects_selection.get_selection(subject_ids)

    results = AutoDict()
    for subject_id in subject_ids:
        for model_name in os.listdir(os.path.join(output_folder, subject_id)):
            info = BatchFitSubjectOutputInfo(os.path.join(output_folder, subject_id, model_name),
                                             subject_id, model_name)
            results[subject_id][model_name] = func(info)
    return results.to_normal_dict()


def collect_batch_fit_single_map(output_folder, collection_folder, model_name, map_name,
                                 subjects_selection=None, symlink=True,
                                 symlink_absolute=False, move=False):
    """Load for every subject and model a single map, name it with the subject id and place it in the collection folder.

    The results are placed in the output folder with as basename the subject id.
    Example: ``<collection_folder>/<subject_id>.nii(.gz)``

    Args:
        output_folder (str): The folder with the results from batch fitting.
        collection_folder (str): The path to the folder where all the files will be put.
        model_name (str): the name of the model for which we want to get the map
        map_name (str): the (base)name of the map we want to retreive (this function will take care of the extension).
        subjects_selection (BatchSubjectSelection): the subjects to use for processing.
            If None all subjects are processed.
        symlink (boolean): only available under Unix OS's. Creates a symlink instead of copying.
            This will create an absolute position symlink.
        symlink_absolute (boolean): if symlink is set to true, do you want an absolute symlink (True)
            or a relative one (False)
        move (boolean): instead of copying the files, move them to a new position. If set, this overrules the parameter
            symlink.
    """
    if not os.path.exists(collection_folder):
        os.makedirs(collection_folder)

    def copy_function(subject_info):
        if subject_info.model_name == model_name:
            map_file = glob.glob(os.path.join(subject_info.output_path, map_name + '.nii*'))[0]
            extension = split_image_path(map_file)[2]

            subject_out = os.path.join(collection_folder, subject_info.subject_id + '{}'.format(extension))

            if os.path.exists(subject_out) or os.path.islink(subject_out):
                if os.path.islink(subject_out):
                    os.unlink(subject_out)
                else:
                    os.remove(subject_out)

            if move:
                shutil.move(map_file, subject_out)
            elif symlink:
                if symlink_absolute:
                    os.symlink(map_file, subject_out)
                else:
                    os.symlink(os.path.relpath(map_file, os.path.dirname(subject_out)), subject_out)
            else:
                shutil.copy(map_file, subject_out)

    run_function_on_batch_fit_output(copy_function, output_folder, subjects_selection=subjects_selection)
