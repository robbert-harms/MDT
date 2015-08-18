import copy
import glob
import re
import distutils.dir_util
import logging
import logging.config as logging_config
import os
import collections
import shutil
import tempfile
from six import string_types
import numpy as np
import nibabel as nib
import pkg_resources
import time
from mdt.components_loader import BatchProfilesLoader, get_model
from mdt.protocols import load_from_protocol
from mdt.cl_routines.mapping.calculate_eigenvectors import CalculateEigenvectors
from mot import runtime_configuration
from mot.base import AbstractProblemData, ModelFunction
import configuration
from mot.cl_environments import CLEnvironmentFactory
from mot.cl_routines.optimizing.meta_optimizer import MetaOptimizer
from mot.factory import get_load_balance_strategy_by_name, get_optimizer_by_name, get_filter_by_name
from scipy.special import jnp_zeros

try:
    import codecs
except ImportError:
    codecs = None

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRIProblemData(AbstractProblemData):

    def __init__(self, prtcl_data_dict, dwi_volume, mask, volume_header):
        """This overrides the standard problem data to also include a mask.

        Args:
            prtcl_data_dict (Protocol): The protocol object used as input data to the model
            dwi_volume (ndarray): The DWI data (4d matrix)
            mask (ndarray): The mask used to create the observations list
            volume_header (nifti header): The header of the nifti file to use for writing the results.

        Attributes:
            dwi_volume (ndarray): The DWI volume
            mask (ndarray): The mask used to create the observations list
            volume_header (nifti header): The header of the nifti file to use for writing the results.
        """
        self.dwi_volume = dwi_volume
        self.mask = mask
        self.volume_header = volume_header
        self._prtcl_data_dict = prtcl_data_dict
        self._observation_list = None

    @property
    def protocol(self):
        """Return the prtcl_data_dict.

        Returns:
            protocol: The protocol object given in the instantiation.
        """
        return self.prtcl_data_dict

    @property
    def prtcl_data_dict(self):
        """Return the constant data stored in this problem data container.

        Returns:
            dict: The protocol data dict.
        """
        return self._prtcl_data_dict

    @property
    def observations(self):
        """Return the constant data stored in this problem data container.

        Returns:
            ndarray: The list of observations
        """
        if self._observation_list is None:
            self._observation_list = create_roi(self.dwi_volume, self.mask)
        return self._observation_list


class DMRICompartmentModelFunction(ModelFunction):

    def __init__(self, name, cl_function_name, parameter_list, cl_header_file, cl_code_file, dependency_list):
        super(DMRICompartmentModelFunction, self).__init__(name, cl_function_name, parameter_list,
                                                           dependency_list=dependency_list)
        self._cl_header_file = cl_header_file
        self._cl_code_file = cl_code_file

    def get_cl_header(self):
        return self._get_cl_dependency_headers() + "\n" + open(os.path.abspath(self._cl_header_file), 'r').read()

    def get_cl_code(self):
        return self._get_cl_dependency_code() + "\n" + open(os.path.abspath(self._cl_code_file), 'r').read()

    def _get_single_dir_coordinate_maps(self, theta, phi, r):
        """Convert spherical coordinates to cartesian coordinates in 3d

        Args:
            theta (ndarray): the double array with the theta values
            phi (ndarray): the double array with the phi values
            r (ndarray): the double array with the r values

        Returns:
            three ndarrays, per vector one map
        """
        cartesian = spherical_to_cartesian(theta, phi)
        extra_dict = {self.name + '.eig0.vec': cartesian, self.name + '.eig0.val': r}

        for ind in range(3):
            extra_dict.update({self.name + '.eig0.vec.' + repr(ind): cartesian[:, ind]})

        return extra_dict


class ProtocolCheckInterface(object):

    def is_protocol_sufficient(self, protocol=None):
        """Check if the protocol holds enough information for this model to work.

        Args:
            protocol (Protocol): The protocol object to check for sufficient information. If set the None, the
                current protocol in the problem data is used.

        Returns:
            boolean: True if there is enough information in the protocol, false otherwise
        """

    def get_protocol_problems(self, protocol=None):
        """Get all the problems with the protocol.

        Args:
            protocol (Protocol): The protocol object to check for problems. If set the None, the
                current protocol in the problem data is used.

        Returns:
            list of ModelProtocolProblem: A list of ModelProtocolProblem instances or subclasses of that baseclass.
                These objects indicate the problems with the protocol and this model.
        """


class ModelProtocolProblem(object):

    def __init__(self):
        """The base class for indicating problems with a protocol.

        These are meant to be returned from the function get_protocol_problems() from the ProtocolCheckInterface.

        Each of these problems is supposed to overwrite the function __str__() for reporting the problem.
        """

    def __repr__(self):
        return self.__str__()

    def can_merge(self, other_problem):
        """If this problem object can merge with the other problem object.

        This can for example always return False if this object can not merge at all. Or can say True to anything
        for merging anything.

        In general it will return True if the problems are of the same class.

        Args:
            other_problem (ModelProtocolProblem): The protocol problem to merge with this one.

        Returns:
            boolean: True if this problem can merge with the other_problem, false otherwise.
        """

    def merge(self, other_problem):
        """Merge another model protocol problem of the same kind into one problem.

        Args:
            other_problem (ModelProtocolProblem): The protocol problem to merge with this one.

        Returns:
            ModelProtocolProblem: A new protocol problem with merged information.
        """


class MissingColumns(ModelProtocolProblem):

    def __init__(self, missing_columns):
        super(MissingColumns, self).__init__()
        self.missing_columns = missing_columns

    def __str__(self):
        return 'Missing columns: ' + ', '.join(self.missing_columns)

    def can_merge(self, other_problem):
        return isinstance(other_problem, MissingColumns)

    def merge(self, other_problem):
        return MissingColumns(self.missing_columns + other_problem.missing_columns)


class InsufficientShells(ModelProtocolProblem):

    def __init__(self, required_nmr_shells, nmr_shells):
        super(InsufficientShells, self).__init__()
        self.required_nmr_shells = required_nmr_shells
        self.nmr_shells = nmr_shells

    def __str__(self):
        return 'Required number of shells is {}, this protocol has {}.'.format(
            self.required_nmr_shells, self.nmr_shells)

    def can_merge(self, other_problem):
        return isinstance(other_problem, InsufficientShells)

    def merge(self, other_problem):
        return InsufficientShells(self.nmr_shells, max(self.required_nmr_shells, other_problem.required_nmr_shells))


class NamedProtocolProblem(ModelProtocolProblem):

    def __init__(self, model_protocol_problem, model_name):
        """This extends the model protocol problem to also include the name of the model.

        Args:
            model_protocol_problem (ModelProtocolProblem): The name for the problem with the given model.
            model_name (str): the name of the model
        """
        super(NamedProtocolProblem, self).__init__()
        self._model_protocol_problem = model_protocol_problem
        self._model_name = model_name

    def __str__(self):
        return "{0}: {1}".format(self._model_name, self._model_protocol_problem)

    def can_merge(self, other_problem):
        return False

    def merge(self, other_problem):
        raise ValueError("This class does not support merging.")


class BatchProfile(object):

    def get_options(self):
        """Get the specific options from this batch fitting profile that will override the default options.

        See the BatchFitting class for the default and available options.
        """

    def get_output_directory(self, root_dir, subject_id):
        """Get the output directory for the subject with the given subject id.

        Args:
            subject_id (str): the subject id for which to get the output directory.

        """

    def get_subjects(self, root_dir):
        """Get the information about all the subjects in the given folder given the root directory.

        Args:
            root_dir (str): the root directory from which to return all the subjects

        Returns:
            dict: a dictionary with as keys 'dwi', 'bval', 'bvec', 'prtcl' and 'mask'.
                Either prtcl of bval and bvec files must be present. mask can always be None
        """

    def profile_suitable(self, root_dir):
        """Check if this directory can be used to load subjects from using this batch fitting profile.

        This is used for auto detecting the best batch fitting profile to use for loading
        subjects from the given root dir.

        Args:
            root_dir (str): the root directory to check if it is usable

        Returns:
            boolean: true if this batch fitting profile can load datasets from this root directory, false otherwise.
        """

    def get_subjects_count(self, root_dir):
        """Get the number of subjects this batch fitting profile can load from the given root directory.

        Args:
            root_dir (str): the root directory to check if it is usable

        Returns:
            int: the number of subjects this batch fitting profile can load from the given directory.
        """


class SimpleBatchProfile(BatchProfile):

    def get_options(self):
        return {}

    def get_output_directory(self, root_dir, subject_id):
        return os.path.join(root_dir, subject_id, 'output')

    def get_subjects(self, root_dir):
        return self._get_subjects(root_dir)

    def profile_suitable(self, root_dir):
        return len(self._get_subjects(root_dir)) > 0

    def get_subjects_count(self, root_dir):
        return len(self._get_subjects(root_dir))

    def _get_subjects(self, root_dir):
        """Get the all the matching subjects from the given root dir.

        Args:
            root_dir (str): the root directory from which to return all the subjects

        Returns:
            list of list: A list of subjects with as first element the subject name and as second the a dictionary with
                as keys 'dwi', 'bval', 'bvec', 'prtcl' and 'mask'.
                Please note, either prtcl or bval and bvec files must be present. The mask can always be None
        """
        return []

    def _get_basename(self, file):
        """Get the basename of a file. That is, the name of the files with all extensions stripped."""
        bn = os.path.basename(file)
        bn = bn.replace('.nii.gz', '')
        exts = os.path.splitext(bn)
        if len(exts) > 0:
            return exts[0]
        return bn


class BatchFitOutputInfo(object):

    def __init__(self, data_folder, batch_profile=None):
        """Single point of information about batch fitting output.

        Args:
            data_folder (str): The data folder with the output files
            batch_profile (BatchProfile or str): the batch profile to use, can also be the name of a batch profile to load.
                If not given it is auto detected.
        """
        self._data_folder = data_folder
        self._batch_profile = get_batch_profile(batch_profile, data_folder)
        self._subjects = self._batch_profile.get_subjects(data_folder)
        self._subjects_dirs = {s_id: self._batch_profile.get_output_directory(data_folder, s_id)
                               for s_id,_ in self._subjects}
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

    def subject_model_path_generator(self, mask_name, error_on_missing_mask=False):
        """Generates for every subject the path to the models for the given mask name.

        If a subject does not have that mask_name it is either skipped or an error is raised, depending on the setting
        error_on_missing_mask.

        Args:
            mask_name (str): the name of the mask we return the path to per subject
            error_on_missing_mask (boolean): if we don't have the mask for one subject should we raise an error or skip
                the subject?

        Returns:
            generator: per subject a tuple: (subject_id, path)
        """
        mask_paths = self.get_path_to_mask_per_subject(mask_name, error_on_missing_mask)
        for subject_id, mask_path in mask_paths.items():
            for d in os.listdir(mask_path):
                full_path = os.path.join(mask_path, d)
                if os.path.isdir(full_path):
                    yield (subject_id, d, full_path)


class PathJoiner(object):

    def __init__(self, *args):
        """The path joining class.

        To construct use something like:
        >>> pjoin = PathJoiner(r'/my/images/dir/')

        or:
        >>> pjoin = PathJoiner('my', 'images', 'dir')


        Then, you can call it like:
        >>> pjoin()
        /my/images/dir

        At least, it returns the above on Linux. On windows it will return 'my\\images\\dir'.

        You can also call it with additional path elements which should be appended to the path:
        >>> pjoin('/brain_mask.nii.gz')
        /my/images/dir/brain_mask.nii.gz

        Note that that is not permanent. To make it permanent you can call
        >>> pjoin.append('results')

        This will extend the stored path to /my/images/dir/results/:
        >>> pjoin('/brain_mask.nii.gz')
        /my/images/dir/results/brain_mask.nii.gz

        You can revert this by calling:
        >>> pjoin.reset()

        You can also create a copy of this class with extended path elements by calling
        >>> pjoin2 = pjoin.create_extended('results')

        This returns a new PathJoiner instance with as path the current path plus the items in the arguments.
        >>> pjoin2('brain_mask.nii.gz')
        /my/images/dir/results/brain_mask.nii.gz

        Args:
            *args: the initial path element(s).
        """
        self._initial_path = os.path.join('', *args)
        self._path = os.path.join('', *args)

    def create_extended(self, *args):
        """Create and return a new PathJoiner instance with the path extended by the given arguments."""
        return PathJoiner(os.path.join(self._path, *args))

    def append(self, *args):
        """Extend the stored path with the given elements"""
        self._path = os.path.join(self._path, *args)
        return self

    def reset(self, *args):
        """Reset the path to the path at construction time"""
        self._path = self._initial_path
        return self

    def __call__(self, *args, **kwargs):
        tmp_path = os.path.join(self._path, *args)
        return os.path.abspath(os.path.join(self._path, tmp_path))


def condense_protocol_problems(protocol_problems_list):
    """Condenses the protocol problems list by combining similar problems objects.

    This uses the function 'merge' from the protocol problems to merge similar items into one.

    Args:
        protocol_problems_list (list of ModelProtocolProblem): the list with the problem objects.

    Returns:
        list of ModelProtocolProblem: A condensed list of the problems
    """
    result_list = []
    protocol_problems_list = list(protocol_problems_list)
    has_merged = False

    for i, mpp in enumerate(protocol_problems_list):
        merged_this = False

        if mpp is not None and mpp:
            for j in range(i + 1, len(protocol_problems_list)):
                for mpp_item in flatten(mpp):
                    if mpp_item.can_merge(protocol_problems_list[j]):
                        result_list.append(mpp_item.merge(protocol_problems_list[j]))
                        protocol_problems_list[j] = None
                        has_merged = True
                        merged_this = True
                        break

            if not merged_this:
                result_list.append(mpp)

    if has_merged:
        return condense_protocol_problems(result_list)
    return list(flatten(result_list))


def split_dataset(dataset, split_dimension, split_index):
    """Split the given dataset along the given dimension on the given index.

    Args:
        dataset (ndarray, list, tuple or dict): The single or list of volume which to split in two
        split_dimension (int): The dimension along which to split the dataset
        split_index (int): The index on the given dimension to split the volume(s)

    Returns:
        If dataset is a single volume return the two volumes that when concatenated give the original volume back.
        If it is a list, tuple or dict return two of those with exactly the same indices but with each holding one half
        of the splitted data.
    """
    if isinstance(dataset, (tuple, list)):
        output_1 = []
        output_2 = []
        for d in dataset:
            split = split_dataset(d, split_dimension, split_index)
            output_1.append(split[0])
            output_2.append(split[1])

        if isinstance(dataset, tuple):
            return tuple(output_1), tuple(output_2)

        return output_1, output_2

    elif isinstance(dataset, dict):
        output_1 = {}
        output_2 = {}
        for k, d in dataset.items():
            split = split_dataset(d, split_dimension, split_index)
            output_1[k] = split[0]
            output_2[k] = split[1]

        return output_1, output_2

    ind_1 = [slice(None)] * dataset.ndim
    ind_1[split_dimension] = range(0, split_index)

    ind_2 = [slice(None)] * dataset.ndim
    ind_2[split_dimension] = range(split_index, dataset.shape[split_dimension])

    return dataset[ind_1], dataset[ind_2]


def get_bessel_roots(number_of_roots=30, np_data_type=np.float64):
    """These roots are used in some of the compartment models. It are the roots of the equation J'_1(x) = 0.

    That is, where J_1 is the first order Bessel function of the first kind.

    Args:
        number_of_root (int): The number of roots we want to calculate.

    Returns:
        ndarray: A vector with the indicated number of bessel roots (of the first order Bessel function
            of the first kind).
    """
    return jnp_zeros(1, number_of_roots).astype(np_data_type, copy=False, order='C')


def read_split_write_volume(volume_fname, first_output_fname, second_output_fname, split_dimension, split_index):
    """Read the given dataset from file, then split it along the given dimension on the given index.

    This writes two files, first_output_fname and second_output_fname
    with respectively the first and second halves of the split dataset.

    Args:
        volume_fname (str): The filename of the volume to load and split
        first_output_fname (str): The filename of the first half of the split
        second_output_fname (str): The filename of the second half of the split
        split_dimension (int): The dimension along which to split the dataset
        split_index (int): The index on the given dimension to split the volume(s)
    """
    signal_img = nib.load(volume_fname)
    signal4d = signal_img.get_data()
    img_header = signal_img.get_header()

    split = split_dataset(signal4d, split_dimension, split_index)

    nib.Nifti1Image(split[0], None, img_header).to_filename(first_output_fname)
    nib.Nifti1Image(split[1], None, img_header).to_filename(second_output_fname)


def create_slice_roi(brain_mask, roi_dimension, roi_slice):
    """Create a region of interest out of the given brain mask by taking one specific slice out of the mask.

    Args:
        brain_mask (ndarray): The brain_mask used to create the new brain mask
        roi_dimension (int): The dimension to take a slice out of
        roi_slice (int): The index on the given dimension.

    Returns:
        A brain mask of the same dimensions as the original mask, but with only one slice activated.
    """
    roi_mask = get_slice_in_dimension(brain_mask, roi_dimension, roi_slice)
    brain_mask = np.zeros_like(brain_mask)

    ind_pos = [slice(None)] * brain_mask.ndim
    ind_pos[roi_dimension] = roi_slice
    brain_mask[tuple(ind_pos)] = roi_mask

    return brain_mask


def concatenate_two_mri_measurements(datasets):
    """ Concatenate the given datasets (combination of signal list and protocols)

    For example, as input one can give:
        ((protocol_1, signal4d_1), (protocol_2, signal4d_2))
    And the expected output is:
        (protocol, signal_list)

    Where the signal_list is for every voxel a concatenation of the given signal lists, and the protocol is a
    concatenation of the given protocols.

    Args:
        datasets: a list of datasets, where a dataset is a tuple structured as: (protocol, signal_list).

    Returns
        A single tuple holding the concatenation of the given datasets
    """
    signal_list = [datasets[0][1]]
    protocol_concat = datasets[0][0].deepcopy()
    for i in range(1, len(datasets)):
        signal_list.append(datasets[i][1])
        protocol_concat.append_protocol(datasets[i][0])
    signal4d_concat = np.concatenate(signal_list, 3)
    return protocol_concat, signal4d_concat


def get_slice_in_dimension(volume, dimension, index):
    """From the given volume get a slice on the given dimension (x, y, z, ...) and then on the given index.

    Args:
        volume (ndarray);: the volume, 3d, 4d or more
        dimension (int): the dimension on which we want a slice
        index (int): the index of the slice

    Returns:
        ndarray: A slice (plane) or hyperplane of the given volume
    """
    ind_pos = [slice(None)] * volume.ndim
    ind_pos[dimension] = index
    array_slice = volume[tuple(ind_pos)]
    return np.squeeze(array_slice)


def simple_parameter_init(model, init_data, exclude_cb=None):
    """Initialize the parameters that are named the same in the model and the init_data from the init_data.

    Args:
        model (AbstractModel); The model with the parameters to initialize
        init_data (dict): The initialize data with as keys parameter names (model.param)
            and as values the maps to initialize to.
        exclude_cb (python function); a python function that can be called to check if an parameter needs to be excluded
            from the simple parameter initialization. This function should accept a key with a model.param name and
            should return true if the parameter should be excluded, false otherwise

    Returns
        None, the initialization happens in place.
    """
    if init_data is not None:
        for key, value in init_data.items():
            if exclude_cb and exclude_cb(key):
                continue

            items = key.split('.')
            if len(items) == 2:
                m, p = items
                cmf = model.cmf(m)
                if cmf and cmf.has_parameter_by_name(p):
                    cmf.init(p, value)


def create_roi(data, brain_mask):
    """Create and return masked data of the given brain volume and mask

    Args:
        data (ndarray): a brain volume with four dimensions (x, y, z, w) where w is the length of the protocol,
            or a list, tuple or dictionary with volumes
        brain_mask (ndarray): the mask indicating the region of interest, dimensions: (x, y, z)

    Returns:
        Signal lists for each of the given volumes. The axis are: (voxels, protocol)
    """
    def creator(v):
        if len(v.shape) < 4:
            v = np.reshape(v, list(v.shape) + [1])
        return np.transpose(np.array([np.extract(brain_mask, v[..., i]) for i in range(v.shape[3])]))

    if isinstance(data, dict):
        return {key: creator(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [creator(value) for value in data]
    elif isinstance(data, tuple):
        return (creator(value) for value in data)
    else:
        return creator(data)


def restore_volumes(data, brain_mask, with_volume_dim=True):
    """Restore the given data to a whole brain volume

    The data can be a list, tuple or dictionary or directly a two dimensional list of data points

    Args:
        data (ndarray): the data as a x dimensional list of voxels, or, a list, tuple, or dict of those voxel lists
        brain_mask (ndarray): the brain_mask which was used to generate the data list
        with_volume_dim (boolean): If true we return values with 4 dimensions. The extra dimension is for
            the volume index. If false we return 3 dimensions.

    Returns:
        Either a single whole volume, a list, tuple or dict of whole volumes, depending on the given data.
        If with_volume_ind_dim is set we return values with 4 dimensions. (x, y, z, 1). If not set we return only
        three dimensions.
    """
    shape3d = brain_mask.shape[:3]
    indices = np.ravel_multi_index(np.nonzero(brain_mask), shape3d[:3], order='C')

    def restorer(voxel_list):
        s = voxel_list.shape

        def restore_3d(voxels):
            volume_length = reduce(lambda x, y: x*y, shape3d[:3])

            return_volume = np.zeros((volume_length,), dtype=voxels.dtype, order='C')
            return_volume[indices] = voxels

            return np.reshape(return_volume, shape3d[:3])

        if len(s) > 1 and s[1] > 1:
            if with_volume_dim:
                volumes = [np.expand_dims(restore_3d(voxel_list[:, i]), axis=3) for i in range(s[1])]
                return np.concatenate(volumes, axis=3)
            else:
                return restore_3d(voxel_list[:, 0])
        else:
            volume = restore_3d(voxel_list)

            if with_volume_dim:
                return np.expand_dims(volume, axis=3)
            return volume

    if isinstance(data, dict):
        return {key: restorer(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [restorer(value) for value in data]
    elif isinstance(data, tuple):
        return (restorer(value) for value in data)
    else:
        return restorer(data)


def spherical_to_cartesian(theta, phi):
    """Convert polar coordinates in 3d space to cartesian unit coordinates.

    x = cos(phi) * sin(theta)
    y = sin(phi) * sin(theta)
    z = cos(theta)

    Args:
        theta (ndarray): The 1d vector with theta's
        phi (ndarray): The 1d vector with phi's

    Returns:
        ndarray: Two dimensional array with on the first axis the voxels and on the second the [x, y, z] coordinates.
    """
    theta = np.squeeze(theta)
    phi = np.squeeze(phi)
    sin_theta = np.sin(theta)
    return np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, np.cos(theta)]).transpose()


def eigen_vectors_from_tensor(theta, phi, psi):
    """Calculate the eigenvectors for a Tensor given the three angles.

    This will return the eigenvectors unsorted, since this function knows nothing about the eigenvalues. The caller
    of this function will have to sort them by eigenvalue if necessary.

    Args:
        theta_roi (ndarray): The list of theta's per voxel in the ROI
        phi_roi (ndarray): The list of phi's per voxel in the ROI
        psi_roi (ndarray): The list of psi's per voxel in the ROI

    Returns:
        The three eigenvectors per voxel in the ROI. The return matrix is of shape (n, 3, 3) where n is the number
        of voxels and the second dimension holds the number of evecs and the last dimension the direction per evec.

        This gives for one voxel the matrix:
            [evec_1_x, evec_1_y, evec_1_z,
             evec_2_x, evec_2_y, evec_2_z
             evec_3_x, evec_3_y, evec_3_z]

        The resulting eigenvectors are the same as those from the Tensor.
    """
    return CalculateEigenvectors(runtime_configuration.runtime_config['cl_environments'],
                                 runtime_configuration.runtime_config['load_balancer']).\
        convert_theta_phi_psi(theta, phi, psi)


def initialize_user_settings(overwrite=True):
    """Initializes the user settings folder using a skeleton.

    This will create all the necessary directories for adding components to MDT. It will also create a basic
    configuration file for setting global wide MDT options.

    If the users home folder already exists a backup copy is created first,

    Args:
        overwrite (boolean): If we want to overwrite the folder if it already exists. If true we overwrite, if false
            we do not.

    Returns:
        the path the user settings skeleton was written to
    """
    path = os.path.join(os.path.expanduser("~"), '.mdt')

    if os.path.exists(path):
        if overwrite:
            backup_dir = os.path.join(path, 'backup_' + time.strftime("%Y-%m-%d"))
            backup_version = 1
            while os.path.isdir(backup_dir):
                backup_dir += '_' + str(backup_version)
                backup_version += 1

            os.mkdir(backup_dir)
            for item in 'components', 'mdt.conf', 'version.txt':
                if os.path.exists(os.path.join(path, item)):
                    shutil.move(os.path.join(path, item), backup_dir)
        else:
            return path

    cache_path = pkg_resources.resource_filename('mdt', 'data/components')
    distutils.dir_util.copy_tree(cache_path, os.path.join(path, 'components'))

    cache_path = pkg_resources.resource_filename('mdt', 'data/mdt.conf')
    shutil.copy(cache_path, path)

    from mdt import __version__
    with open(os.path.join(path, 'version.txt'), 'w') as f:
        f.write(__version__)

    return path


def check_user_components():
    """Check if the components in the user's home folder are up to date with this version of MDT

    Returns:
        bool: True if the .mdt folder exists and the versions are up to date, False otherwise.
    """
    version_file = os.path.join(os.path.expanduser("~"), '.mdt', 'version.txt')
    if not os.path.isfile(version_file):
        return False

    from mdt import __version__
    with open(version_file, 'r') as f:
        version = f.read()
        return version.strip() == __version__.strip()


def setup_logging(disable_existing_loggers=None):
    """Setup global logging.

    This uses the loaded config settings to set up the logging.

    Args:
        disable_existing_loggers (boolean): If we would like to disable the existing loggers when creating this one.
            None means use the default from the config, True and False overwrite the config.
    """
    conf = configuration.config['logging']['info_dict']
    if disable_existing_loggers is not None:
        conf['disable_existing_loggers'] = True

    logging_config.dictConfig(conf)


def configure_per_model_logging(output_path):
    """Set up logging for one specific model.

    Args:
        output_path: the output path where the model results are stored.
    """
    if output_path is None:
        ModelOutputLogHandler.output_file = None
    else:
        ModelOutputLogHandler.output_file = os.path.abspath(os.path.join(output_path, 'info.log'))


class ModelOutputLogHandler(logging.StreamHandler):

    output_file = None

    def __init__(self, mode='a', encoding=None):
        """This logger can log information about a model optimization to the folder of the model being optimized.

        One can change the class attribute 'output_file' to change the file items are logged to.
        """
        super(ModelOutputLogHandler, self).__init__()

        if self.output_file is None:
            self.output_file = tempfile.mkstemp()[1]

        if codecs is None:
            encoding = None
        self.baseFilename = os.path.abspath(self.output_file)
        self.mode = mode
        self.encoding = encoding
        self.stream = None
        self._open()

    def emit(self, record):
        if self.output_file is not None:
            if os.path.abspath(self.output_file) != self.baseFilename:
                self.close()
                self.baseFilename = os.path.abspath(self.output_file)

            if self.stream is None:
                self.stream = self._open()

            super(ModelOutputLogHandler, self).emit(record)

    def close(self):
        """Closes the stream."""
        self.acquire()
        try:
            if self.stream:
                self.flush()
                if hasattr(self.stream, "close"):
                    self.stream.close()
                self.stream = None
            super(ModelOutputLogHandler, self).close()
        finally:
            self.release()

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        if self.encoding is None:
            stream = open(self.baseFilename, self.mode)
        else:
            stream = codecs.open(self.baseFilename, self.mode, self.encoding)
        return stream


def recursive_merge_dict(dictionary, update_dict):
    """ Recursively merge the given dictionary with the new values.

    This does not merge in place, a new dictionary is returned.

    Args:
        dictionary (dict): the dictionary we want to update
        update_dict (dict): the dictionary with the new values

    Returns:
        dict: a combination of the two dictionaries in which the values of the last dictionary take precedence over
            that of the first.
            Example:
                recursive_merge_dict(
                    {'k1': {'k2': 2}},
                    {'k1': {'k2': {'k3': 3}}, 'k4': 4}
                )

                gives:

                {'k1': {'k2': {'k3': 3}}, 'k4': 4}
    """
    dictionary = copy.deepcopy(dictionary)

    def merge(d, upd):
        for k, v in upd.iteritems():
            if isinstance(v, collections.Mapping):
                r = merge(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = upd[k]
        return d

    return merge(dictionary, update_dict)


def load_problem_data(volume_info, protocol, mask):
    """Load and create the problem data object that can be given to a model

    Args:
        volume_info (string): Either an (ndarray, img_header) tuple or the full path to the volume (4d signal data).
        protocol (Protocol or string): A protocol object with the right protocol for the given data,
            or a string object with a filename to the given file.
        mask (string): A full path to a mask file that can optionally be used. If None given, no mask is used.

    Returns:
        The Problem data, in the ProblemData container object.
    """
    if isinstance(protocol, string_types):
        protocol = load_from_protocol(protocol)

    if isinstance(mask, string_types):
        mask = nib.load(mask).get_data() > 0
    if isinstance(volume_info, string_types):
        signal4d, img_header = load_dwi(volume_info)
    else:
        signal4d, img_header = volume_info

    return DMRIProblemData(protocol, signal4d, mask, img_header)


def load_dwi(volume_fname):
    """Load the diffusion weighted image data from the given volume filename.

    This does not perform any data type changes, so the input may not be in float64. If you call this function
    to satisfy load_problem_data() then this is not a problem.

    Args:
        volume_fname (string): The filename of the volume to load.

    Returns:
        a tuple with (data, header) for the given file.
    """
    info = nib.load(volume_fname)
    header = info.get_header()
    data = info.get_data()
    if len(data.shape) < 4:
        data = np.expand_dims(data, axis=3)
    return data, header


def get_batch_profile(batch_profile, data_folder):
    """Wrapper function for getting a batch profile.

    Args:
        batch_profile (None, string or batch profile): indication of the batch profile to load
        data_folder (str): the data folder we want to use the batch profile on.

    Returns:
        If the given batch profile is None we return the output from get_best_batch_profile(). If batch profile is
        a string we load it from the batch profiles loader. Else we return the input.
    """
    if batch_profile is None:
        return get_best_batch_profile(data_folder)
    elif isinstance(batch_profile, string_types):
        return BatchProfilesLoader().load(batch_profile)
    return batch_profile


def get_best_batch_profile(directory):
    """Get the batch profile that best matches the given directory.

    Args:
        directory (str): the directory for which to get the best batch profile.

    Returns:
        BatchProfile: the best matching batch profile.
    """
    profile_loader = BatchProfilesLoader()
    crawlers = [profile_loader.load(c) for c in profile_loader.list_all()]

    best_crawler = None
    best_subjects_count = 0
    for crawler in crawlers:
        if crawler.profile_suitable(directory):
            tmp_count = crawler.get_subjects_count(directory)
            if tmp_count > best_subjects_count:
                best_crawler = crawler
                best_subjects_count = tmp_count

    return best_crawler


def load_brain_mask(brain_mask_fname):
    """Load the brain mask from the given file.

    Args:
        brain_mask_fname (string): The filename of the brain mask to load.

    Returns:
        The loaded brain mask data
    """
    return nib.load(brain_mask_fname).get_data() > 0


def flatten(input_it):
    """Flatten an iterator with a new iterator

    Args:
        it (iterable): the input iterable to flatten

    Returns:
        a new iterable with a flattened version of the original iterable.
    """
    try:
        it = iter(input_it)
    except TypeError:
        yield input_it
    else:
        for i in it:
            for j in flatten(i):
                yield j


class ProtocolProblemError(Exception):
    pass


class MetaOptimizerBuilder(object):

    def __init__(self, meta_optimizer_config=None):
        """Create a new meta optimizer builder.

        This will create a new MetaOptimizer using settings from the config file or from the meta_optimizer_config
        parameter in this constructor.

        If meta_optimizer_config is set it takes precedence over the values in the configuration.

        Args;
            meta_optimizer_config (dict): optimizer configuration settings
        """
        self._meta_optimizer_config = meta_optimizer_config or {}

    def construct(self, model_name=None):
        """Construct a new meta optimizer with the options from the current configuration.

        If model_name is given, we try to load the specific options for that model from the configuration. If it it not
        given we load the general options under 'general/meta_optimizer'.

        Args:
            model_name (str): the name of the model we will optimize with the returned optimizer.
        """
        optim_config = self._get_configuration_dict(model_name)
        meta_optimizer = MetaOptimizer(runtime_configuration.runtime_config['cl_environments'],
                                       runtime_configuration.runtime_config['load_balancer'])

        meta_optimizer.optimizer = self._get_optimizer(optim_config['optimizers'][0])
        meta_optimizer.extra_optim_runs_optimizers = [self._get_optimizer(optim_config['optimizers'][i])
                                                      for i in range(1, len(optim_config['optimizers']))]

        for attr in ('extra_optim_runs', 'extra_optim_runs_apply_smoothing', 'extra_optim_runs_use_perturbation',
                     'enable_grid_search'):
            meta_optimizer.__setattr__(attr, optim_config[attr])

        if 'smoothing_routines' in optim_config and len(optim_config['smoothing_routines']):
            meta_optimizer.smoother = self._get_smoother(optim_config['smoothing_routines'][0])
            meta_optimizer.extra_optim_runs_smoothers = [self._get_smoother(optim_config['smoothing_routines'][i])
                                                         for i in range(1, len(optim_config['smoothing_routines']))]

        if 'load_balancer' in optim_config:
            load_balancer = get_load_balance_strategy_by_name(optim_config['load_balancer']['name'])()
            for attr, value in optim_config['load_balancer'].items():
                if attr != 'name':
                    load_balancer.__setattr__(attr, value)
            meta_optimizer.load_balancer = load_balancer

        return meta_optimizer

    def _get_configuration_dict(self, model_name):
        current_config = configuration.config['optimization_settings']
        optim_config = current_config['general']

        if model_name is not None and 'single_model' in current_config:
            info_dict = get_model_config(model_name, current_config['single_model'])
            if info_dict:
                optim_config = recursive_merge_dict(optim_config, info_dict)

        optim_config = recursive_merge_dict(optim_config, self._meta_optimizer_config)
        return optim_config

    def _get_optimizer(self, options):
        optimizer = get_optimizer_by_name(options['name'])
        patience = None
        if 'patience' in options:
            patience = options['patience']
        return optimizer(runtime_configuration.runtime_config['cl_environments'],
                         runtime_configuration.runtime_config['load_balancer'],
                         patience=patience)

    def _get_smoother(self, options):
        smoother = get_filter_by_name(options['name'])
        size = 1
        if 'size' in options:
            size = options['size']
        return smoother(size,
                        runtime_configuration.runtime_config['cl_environments'],
                        runtime_configuration.runtime_config['load_balancer'])


def collect_batch_fit_output(data_folder, output_dir, batch_profile=None, mask_name=None, symlink=False):
    """Load from the given data folder all the output files and put them into the output directory.

    If there is more than one mask file available the user has to choose which mask to use using the mask_name
    keyword argument. If it is not given an error is raised.

    The results for the chosen mask it placed in the output folder per subject. Example:
        <output_dir>/<subject_id>/<results>

    Args:
        data_folder (str): The data folder with the output files
        output_dir (str): The path to the output folder where all the files will be put.
        batch_profile (BatchProfile or str): the batch profile to use, can also be the name of a batch profile to load.
            If not given it is auto detected.
        mask_name (str): the mask to use to get the output from
        symlink (boolean): only available under Unix OS's. Creates a symlink instead of copying.
    """
    output_info = BatchFitOutputInfo(data_folder, batch_profile)
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


def run_function_on_batch_fit_output(data_folder, func, batch_profile=None, mask_name=None):
    """Run a function on the output of a batch fitting routine.

    This enables you to run a function on every model output from every subject. The python function should accept
    the following arguments in this order:
        - path: the full path to the directory with the maps
        - subject_id: the id of the subject
        - mask_name: the name of the mask
        - model_name: the name of the model

    Args:
        data_folder (str): The data folder with the output files
        func (python function): the python function we should call for every map and model
        batch_profile (BatchProfile or str): the batch profile to use, can also be the name of a batch profile to load.
            If not given it is auto detected.
        mask_name (str): the mask to use to get the output from
    """
    output_info = BatchFitOutputInfo(data_folder, batch_profile)
    mask_names = output_info.get_available_masks()
    for mask_name in mask_names:
        for subject_id, model_name, path in output_info.subject_model_path_generator(mask_name):
            func(path, subject_id, mask_name, model_name)


def get_cl_devices():
    """Get a list of all CL devices in the system.

    The indices of the devices can be used in the model fitting/sampling functions for 'cl_device_ind'.

    Returns:
        A list of CLEnvironments, one for each device in the system.
    """
    return CLEnvironmentFactory.all_devices()


def get_model_config(model_name, config_list):
    """Get from the given dictionary the config for the given model.

    The config list should contain dictionaries with the items 'model_name' and 'config'. Where the first is a regex
    expression for the model name and the second the configuration we will use.

    Args:
        model_name (str): the name of the model
        config_list (list of dict): the list with config items with the keys 'model_name' and 'config'.

    Returns:
        An accumulation of all the configuration of all the models that match with the given model name.
    """
    if not config_list:
        return {}
    conf = {}
    for info in config_list:
        if re.match(info['model_name'], model_name):
            conf = recursive_merge_dict(conf, info['config'])
    return conf


def apply_model_protocol_options(model_protocol_options, problem_data):
    """Apply the model specific protocol options.

    This will check the configuration if there are model specific options for the protocol/DWI data. If so, we
    will create and return a new problem data object. If not so, we will return the old one.

    Args:
        model_protocol_options (dict): a dictionary with the model protocol options to apply to this problem data
        problem_data (DMRIProblemData): the problem data object to which the protocol options are applied

    Returns:
        a new problem data object with the correct protocol (and DWI data), or the old one
    """
    if model_protocol_options:
        protocol = problem_data.protocol
        protocol_indices = np.array([])

        if 'use_weighted' not in model_protocol_options or\
            ('use_weighted' in model_protocol_options and model_protocol_options['use_weighted']):
            if 'b_value' in model_protocol_options:
                options = {'start': 0, 'end': 1e9, 'epsilon': None}
                for key, value in model_protocol_options['b_value'].items():
                    options.update({key: value})
                protocol_indices = protocol.get_indices_bval_in_range(**options)

        if 'use_unweighted' not in model_protocol_options or\
            ('use_unweighted' in model_protocol_options and model_protocol_options['use_unweighted']):
            protocol_indices = np.append(protocol_indices, protocol.get_unweighted_indices())

        protocol_indices = np.unique(protocol_indices)

        if len(protocol_indices) != protocol.protocol_length:
            logger = logging.getLogger(__name__)
            logger.info('Applying model protocol options. We will only use a subset of the protocol.')
            new_protocol = protocol.get_new_protocol_with_indices(protocol_indices)

            new_dwi_volume = problem_data.dwi_volume[..., protocol_indices]

            return DMRIProblemData(new_protocol, new_dwi_volume, problem_data.mask,
                                   problem_data.volume_header)

    return problem_data


def model_output_exists(model, output_folder, check_sample_output=False):
    """Checks if the output for the given model exists in the given output folder.

    This will check for a given model if the output folder exists and contains a nifti file for each parameter
    of the model.

    When using this to try to skip subjects when batch fitting it might fail if one of the models can not be calculated
    for a given subject. For example Noddi requires two shells. If that is not given we can not calculate it and
    hence no maps will be generated. When we are testing if the output exists it will therefore return False.

    Args:
        model (AbstractModel, CascadeModel or str): the model to check for existence, accepts cascade models.
            If a string is given the model is tried to be loaded from the components loader.
        output_folder (str): the folder where the output folder of the results should reside in
        check_sample_output (boolean): if True we also check if there is a subdir 'samples' that contains sample
            results for the given model

    Returns:
        boolean: true if the output folder exists and contains files for all the parameters of the model.
            For a cascade model it returns true if the maps of all the models exist.
    """
    if isinstance(model, string_types):
        model = get_model(model)

    from mdt.cascade_model import CascadeModelInterface
    if isinstance(model, CascadeModelInterface):
        return all([model_output_exists(sub_model, output_folder, check_sample_output)
                    for sub_model in model.get_model_names()])

    output_path = os.path.join(output_folder, model.name)
    parameter_names = model.get_optimization_output_param_names()

    if not os.path.exists(output_path):
        return False

    for parameter_name in parameter_names:
        if not glob.glob(os.path.join(output_path, parameter_name + '*')):
            return False

    if check_sample_output:
        if not os.path.exists(os.path.join(output_path, 'samples', 'samples.pyobj')):
            return False

    return True
