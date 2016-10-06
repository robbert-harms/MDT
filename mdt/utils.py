import collections
import distutils.dir_util
import glob
import logging
import logging.config as logging_config
import os
import re
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager

import nibabel as nib
import numpy as np
import pkg_resources
import six
from numpy.lib.format import open_memmap
from scipy.special import jnp_zeros
from six import string_types

import mot.utils
from mdt.IO import Nifti
from mdt.cl_routines.mapping.calculate_eigenvectors import CalculateEigenvectors
from mdt.components_loader import get_model
from mdt.configuration import get_config_dir
from mdt.configuration import get_logging_configuration_dict, get_noise_std_estimators, get_tmp_results_dir
from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
from mdt.data_loaders.noise_std import autodetect_noise_std_loader
from mdt.data_loaders.protocol import autodetect_protocol_loader
from mdt.exceptions import NoiseStdEstimationNotPossible
from mdt.log_handlers import ModelOutputLogHandler
from mdt.protocols import load_protocol, write_protocol
from mot.model_building.problem_data import AbstractProblemData
from mot.cl_environments import CLEnvironmentFactory
from mot.cl_routines.mapping.calculate_model_estimates import CalculateModelEstimates
from mot.cl_routines.mapping.loglikelihood_calculator import LogLikelihoodCalculator
from mot.model_building.evaluation_models import OffsetGaussianEvaluationModel

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

    def __init__(self, protocol, dwi_volume, mask, volume_header, static_maps=None, gradient_deviations=None,
                 noise_std=None):
        """An implementation of the problem data for diffusion MRI models.

        Args:
            protocol (Protocol): The protocol object used as input data to the model
            dwi_volume (ndarray): The DWI data (4d matrix)
            mask (ndarray): The mask used to create the observations list
            volume_header (nifti header): The header of the nifti file to use for writing the results.
            static_maps (Dict[str, ndarray]): the static maps used as values for the static map parameters
            gradient_deviations (ndarray): the gradient deviations containing per voxel 9 values that constitute the
                gradient non-linearities. Of the 4d matrix the first 3 dimensions are supposed to be the voxel
                index and the 4th should contain the grad dev data.
            noise_std (number or ndarray): either None for automatic detection,
                or a scalar, or an 3d matrix with one value per voxel.

        Attributes:
            dwi_volume (ndarray): The DWI volume
            volume_header (nifti header): The header of the nifti file to use for writing the results.
        """
        self._logger = logging.getLogger(__name__)
        self.dwi_volume = dwi_volume
        self.volume_header = volume_header
        self._mask = mask
        self._protocol = protocol
        self._observation_list = None
        self._static_maps = static_maps or {}
        self.gradient_deviations = gradient_deviations
        self._noise_std = noise_std

    def copy_with_updates(self, *args, **kwargs):
        """Create a copy of this problem data, while setting some of the arguments to new values.

        You can use any of the arguments (args and kwargs) of the constructor for this call.
        If given we will use those values instead of the values in this problem data object for the copy.
        """
        new_args = [self._protocol, self.dwi_volume, self._mask, self.volume_header]
        for ind, value in enumerate(args):
            new_args[ind] = value

        new_kwargs = dict(static_maps=self._static_maps, gradient_deviations=self.gradient_deviations,
                          noise_std=self._noise_std)
        for key, value in kwargs.items():
            new_kwargs[key] = value

        return DMRIProblemData(*new_args, **new_kwargs)

    def get_nmr_inst_per_problem(self):
        return self._protocol.length

    @property
    def protocol(self):
        return self._protocol

    @property
    def observations(self):
        if self._observation_list is None:
            self._observation_list = create_roi(self.dwi_volume, self._mask)
        return self._observation_list

    @property
    def mask(self):
        """Return the mask in use

        Returns:
            np.array: the numpy mask array
        """
        return self._mask

    @mask.setter
    def mask(self, new_mask):
        """Set the new mask and update the observations list.

        Args:
            new_mask (np.array): the new mask
        """
        self._mask = new_mask
        self._observation_list = None

    @property
    def static_maps(self):
        """Get the static maps. They are used as data for the static parameters.

        Returns:
            Dict[str, val]: per static map the value for the static map. This can either be an one or two dimensional
                matrix containing the values for each problem instance or it can be a single value we will use
                for all problem instances.
        """
        if self._static_maps is not None:
            return_items = {}

            for key, val in self._static_maps.items():

                loaded_val = None

                if isinstance(val, six.string_types):
                    loaded_val = create_roi(load_nifti(val).get_data(), self.mask)
                elif isinstance(val, np.ndarray):
                    loaded_val = create_roi(val, self.mask)
                elif is_scalar(val):
                    loaded_val = val

                return_items[key] = loaded_val

            return return_items

        return self._static_maps

    @property
    def noise_std(self):
        """The noise standard deviation we will use during model evaluation.

        During optimization or sampling the model will be evaluated against the observations using an evaluation
        model. Most of these evaluation models need to have a standard deviation.

        Returns:
            number of ndarray: either a scalar or a 2d matrix with one value per problem instance.
        """
        try:
            noise_std = autodetect_noise_std_loader(self._noise_std).get_noise_std(self)
        except NoiseStdEstimationNotPossible:
            logger = logging.getLogger(__name__)
            logger.warn('Failed to obtain a noise std for this subject. We will continue with an std of 1.')
            noise_std = 1

        if is_scalar(noise_std):
            self._logger.info('Using a scalar noise standard deviation of {0}'.format(noise_std))
            return noise_std
        else:
            self._logger.info('Using a voxel wise noise standard deviation.')
            return create_roi(noise_std, self.mask)


class MockDMRIProblemData(DMRIProblemData):

    def __init__(self, protocol=None, dwi_volume=None, mask=None, volume_header=None,
                 **kwargs):
        """A mock DMRI problem data object that returns None for everything unless given.
        """
        super(MockDMRIProblemData, self).__init__(protocol, dwi_volume, mask, volume_header, **kwargs)

    @property
    def observations(self):
        return self._observation_list

    @property
    def noise_std(self):
        return self._noise_std


class PathJoiner(object):

    def __init__(self, *args):
        """The path joining class.

        To construct use something like:

        .. code-block:: python

            >>> pjoin = PathJoiner(r'/my/images/dir/')

        or:

        .. code-block:: python

            >>> pjoin = PathJoiner('my', 'images', 'dir')


        Then, you can call it like:

        .. code-block:: python

            >>> pjoin()
            /my/images/dir


        At least, it returns the above on Linux. On windows it will return ``my\\images\\dir``. You can also call it
        with an additional path element that is (temporarily) appended to the path:

        .. code-block:: python

            >>> pjoin('/brain_mask.nii.gz')
            /my/images/dir/brain_mask.nii.gz

        To add a path permanently to the path joiner use:

        .. code-block:: python

            >>> pjoin.append('results')

        This will extend the stored path to ``/my/images/dir/results/``:

        .. code-block:: python

            >>> pjoin('/brain_mask.nii.gz')
            /my/images/dir/results/brain_mask.nii.gz

        You can reset the path joiner to the state of at object construction using:

        .. code-block:: python

            >>> pjoin.reset()

        You can also create a copy of this class with extended path elements by calling

        .. code-block:: python

            >>> pjoin2 = pjoin.create_extended('results')

        This returns a new PathJoiner instance with as path the current path plus the items in the arguments.

        .. code-block:: python

            >>> pjoin2('brain_mask.nii.gz')
            /my/images/dir/results/brain_mask.nii.gz

        Args:
            *args: the initial path element(s).
        """
        self._initial_path = os.path.abspath(os.path.join('', *args))
        self._path = os.path.abspath(os.path.join('', *args))

    def create_extended(self, *args):
        """Create and return a new PathJoiner instance with the path extended by the given arguments."""
        return PathJoiner(os.path.join(self._path, *args))

    def append(self, *args):
        """Extend the stored path with the given elements"""
        self._path = os.path.join(self._path, *args)
        return self

    def reset(self):
        """Reset the path to the path at construction time"""
        self._path = self._initial_path
        return self

    def make_dirs(self, mode=0o777):
        """Create the directories if they do not exists.

        This uses os.makedirs to make the directories. The given argument mode is handed to os.makedirs.

        Args:
            mode: the mode parameter for os.makedirs
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path, mode)

    def __call__(self, *args):
        return os.path.abspath(os.path.join(self._path, *args))


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


def split_write_dataset(input_fname, split_dimension, split_index, output_folder=None):
    """Split the given dataset using the function split_dataset and write the output files.

    Args:
        dataset (str): The filename of a volume to split
        split_dimension (int): The dimension along which to split the dataset
        split_index (int): The index on the given dimension to split the volume(s)
    """
    if output_folder is None:
        output_folder = os.path.dirname(input_fname)

    dataset = load_nifti(input_fname)
    data = dataset.get_data()

    split = split_dataset(data, split_dimension, split_index)

    basename = os.path.basename(input_fname).split('.')[0]
    length = data.shape[split_dimension]
    lengths = (repr(0) + 'to' + repr(split_index-1), repr(split_index) + 'to' + repr(length-1))

    volumes = {}
    for ind, v in enumerate(split):
        volumes.update({str(basename) + '_split_' + str(split_dimension) + '_' + lengths[ind]: v})

    Nifti.write_volume_maps(volumes, output_folder, dataset.get_header())


def get_bessel_roots(number_of_roots=30, np_data_type=np.float64):
    """These roots are used in some of the compartment models. It are the roots of the equation ``J'_1(x) = 0``.

    That is, where ``J_1`` is the first order Bessel function of the first kind.

    Args:
        number_of_root (int): The number of roots we want to calculate.
        np_data_type (np.data_type): the numpy data type

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
        volume_fname (str): The filename of the volume to use and split
        first_output_fname (str): The filename of the first half of the split
        second_output_fname (str): The filename of the second half of the split
        split_dimension (int): The dimension along which to split the dataset
        split_index (int): The index on the given dimension to split the volume(s)
    """
    signal_img = load_nifti(volume_fname)
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
    roi_mask = np.zeros_like(brain_mask)

    ind_pos = [slice(None)] * roi_mask.ndim
    ind_pos[roi_dimension] = roi_slice
    roi_mask[tuple(ind_pos)] = get_slice_in_dimension(brain_mask, roi_dimension, roi_slice)

    return roi_mask


def write_slice_roi(brain_mask_fname, roi_dimension, roi_slice, output_fname, overwrite_if_exists=False):
    """Create a region of interest out of the given brain mask by taking one specific slice out of the mask.

    This will both write and return the created slice ROI.

    We need a filename as input brain mask since we need the header of the file to be able to write the output file
    with the same header.

    Args:
        brain_mask_fname (string): The filename of the brain_mask used to create the new brain mask
        roi_dimension (int): The dimension to take a slice out of
        roi_slice (int): The index on the given dimension.
        output_fname (string): The output filename
        overwrite_if_exists (boolean, optional, default false): If we want to overwrite the file if it already exists

    Returns:
        A brain mask of the same dimensions as the original mask, but with only one slice set to one.
    """
    if os.path.exists(output_fname) and not overwrite_if_exists:
        return load_brain_mask(output_fname)

    if not os.path.isdir(os.path.dirname(output_fname)):
        os.makedirs(os.path.dirname(output_fname))

    brain_mask_img = load_nifti(brain_mask_fname)
    brain_mask = brain_mask_img.get_data()
    img_header = brain_mask_img.get_header()
    roi_mask = create_slice_roi(brain_mask, roi_dimension, roi_slice)
    nib.Nifti1Image(roi_mask, None, img_header).to_filename(output_fname)
    return roi_mask


def concatenate_two_mri_measurements(datasets):
    """ Concatenate the given datasets (combination of volumes and protocols)

    For example, as input one can give:

    .. code-block:: python

        ((protocol_1, volume_1),
         (protocol_2, volume_2),
         ...)

    And the output is: ``(protocol, volumes)`` where the volumes is for every voxel a concatenation
    of the given volumes, and the protocol is a concatenation of the given protocols.

    Args:
        datasets: a list of datasets, where a dataset is a tuple structured as: (protocol, volume).

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
        volume (ndarray): the volume, 3d, 4d or more
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

    This initializes the parameters in place.

    Args:
        model (AbstractModel); The model with the parameters to initialize
        init_data (dict): The initialize data with as keys parameter names (model.param)
            and as values the maps to initialize to.
        exclude_cb (python function); a python function that can be called to check if an parameter needs to be excluded
            from the simple parameter initialization. This function should accept a key with a model.param name and
            should return true if the parameter should be excluded, false otherwise
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
        data (string, ndarray or dict): a brain volume with four dimensions (x, y, z, w)
            where w is the length of the protocol, or a list, tuple or dictionary with volumes or a string
            with a filename of a dataset to use.
        brain_mask (ndarray or str): the mask indicating the region of interest with dimensions: (x, y, z) or the string
            to the brain mask to use

    Returns:
        ndarray, tuple, dict: If a single ndarray is given we will return the ROI for that array. If
            an iterable is given we will return a tuple. If a dict is given we return a dict.
            For each result the axis are: (voxels, protocol)
    """
    from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
    brain_mask = autodetect_brain_mask_loader(brain_mask).get_data()

    def creator(v):
        return_val = v[brain_mask]
        if len(return_val.shape) == 1:
            return_val = np.expand_dims(return_val, axis=1)
        return return_val

    if isinstance(data, dict):
        return DeferredROICreationDict(data, brain_mask)
    elif isinstance(data, (list, tuple)):
        return DeferredROICreationTuple(data, brain_mask)
    elif isinstance(data, six.string_types):
        return creator(load_nifti(data).get_data())
    return creator(data)


class DeferredROICreationDict(collections.MutableMapping):

    def __init__(self, items, mask):
        """Deferred ROI creation of the given items using the given mask.

        On the moment one of the keys of this dict class is requested we will create the ROI if it does not yet exists.
        The advantage of this class is that it saves memory by deferring the loading until an item is really needed.

        Args:
            items (dict): the items we want to create a ROI of
            mask (ndarray): the matrix we use for creating the ROI
        """
        self._items = items.copy()
        self._mask = mask
        self._computed_rois = {}

    def __delitem__(self, key):
        del self._items[key]
        if key in self._computed_rois:
            del self._computed_rois[key]

    def __getitem__(self, key):
        if key not in self._computed_rois:
            self._computed_rois[key] = create_roi(self._items[key], self._mask)
        return self._computed_rois[key]

    def __contains__(self, key):
        try:
            self._items[key]
        except KeyError:
            return False
        else:
            return True

    def __iter__(self):
        for key in self._items.keys():
            yield key

    def __len__(self):
        return len(self._items)

    def __setitem__(self, key, value):
        self._computed_rois[key] = value


class DeferredROICreationTuple(collections.Sequence):

    def __init__(self, items, mask):
        """Deferred ROI creation of the given items using the given mask.

        On the moment one of the items of this class is requested we will create the ROI if it does not yet exists.
        The advantage of this class is that it saves memory by deferring the loading until an item is really needed.

        Args:
            items (list, tuple): the items we want to create a ROI of
            mask (ndarray): the matrix we use for creating the ROI
        """
        self._items = items
        self._mask = mask
        self._computed_rois = {}

    def __getitem__(self, index):
        if index not in self._computed_rois:
            self._computed_rois[index] = create_roi(self._items[index], self._mask)
        return self._computed_rois[index]

    def __len__(self):
        return len(self._items)


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
    from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
    brain_mask = autodetect_brain_mask_loader(brain_mask).get_data()

    shape3d = brain_mask.shape[:3]
    indices = np.ravel_multi_index(np.nonzero(brain_mask)[:3], shape3d, order='C')

    def restorer(voxel_list):
        s = voxel_list.shape

        def restore_3d(voxels):
            return_volume = np.zeros((brain_mask.size,), dtype=voxels.dtype, order='C')
            return_volume[indices] = voxels
            return np.reshape(return_volume, shape3d)

        def restore_4d(voxels):
            return_volume = np.zeros((brain_mask.size, s[1]), dtype=voxels.dtype, order='C')
            return_volume[indices] = voxels
            return np.reshape(return_volume, brain_mask.shape + (s[1], ))

        if len(s) > 1 and s[1] > 1:
            if with_volume_dim:
                return restore_4d(voxel_list)
            else:
                return restore_3d(voxel_list[:, 0])
        else:
            volume = restore_3d(voxel_list)

            if with_volume_dim:
                return np.expand_dims(volume, axis=3)
            return volume

    if isinstance(data, collections.MutableMapping):
        return {key: restorer(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [restorer(value) for value in data]
    elif isinstance(data, tuple):
        return (restorer(value) for value in data)
    elif isinstance(data, collections.Sequence):
        return [restorer(value) for value in data]
    else:
        return restorer(data)


def spherical_to_cartesian(theta, phi):
    """Convert polar coordinates in 3d space to cartesian unit coordinates.

    .. code-block:: python

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
    return_val = np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, np.cos(theta)]).transpose()

    if len(return_val.shape) == 1:
        return return_val[np.newaxis, :]

    return return_val


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
        of voxels and the three by three matrix is:

        .. code-block:: python

            [[evec_1_x, evec_1_y, evec_1_z],
             [evec_2_x, evec_2_y, evec_2_z],
             [evec_3_x, evec_3_y, evec_3_z]]

        The resulting eigenvectors are the same as those from the Tensor.
    """
    return CalculateEigenvectors().convert_theta_phi_psi(theta, phi, psi)


def init_user_settings(pass_if_exists=True):
    """Initializes the user settings folder using a skeleton.

    This will create all the necessary directories for adding components to MDT. It will also create a basic
    configuration file for setting global wide MDT options. Also, it will copy the user components from the previous
    version to this version.

    Each MDT version will have it's own sub-directory in the config directory.

    Args:
        pass_if_exists (boolean): if the folder for this version already exists, we might do nothing (if True)

    Returns:
        str: the path the user settings skeleton was written to
    """
    from mdt.configuration import get_config_dir
    path = get_config_dir()
    base_path = os.path.dirname(get_config_dir())

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    @contextmanager
    def tmp_save_latest_version():
        versions_available = list(reversed(sorted(os.listdir(base_path))))
        tmp_dir = tempfile.mkdtemp()

        if versions_available:
            previous_version = versions_available[0]

            if os.path.exists(os.path.join(base_path, previous_version, 'components', 'user')):
                shutil.copytree(os.path.join(base_path, previous_version, 'components', 'user'),
                                tmp_dir + '/components/')

            if os.path.isfile(os.path.join(base_path, previous_version, 'mdt.conf')):
                shutil.copy(os.path.join(base_path, previous_version, 'mdt.conf'), tmp_dir + '/mdt.conf')

            if os.path.isfile(os.path.join(base_path, previous_version, 'mdt.gui.conf')):
                shutil.copy(os.path.join(base_path, previous_version, 'mdt.gui.conf'), tmp_dir + '/mdt.gui.conf')

        yield tmp_dir
        shutil.rmtree(tmp_dir)

    def init_from_mdt():
        cache_path = pkg_resources.resource_filename('mdt', 'data/components')
        distutils.dir_util.copy_tree(cache_path, os.path.join(path, 'components'))

        cache_path = pkg_resources.resource_filename('mdt', 'data/mdt.conf')
        shutil.copy(cache_path, path + '/mdt.default.conf')

        if not os.path.exists(path + '/components/user/'):
            os.makedirs(path + '/components/user/')

    def copy_user_components(tmp_dir):
        if os.path.exists(tmp_dir + '/components/'):
            shutil.rmtree(os.path.join(path, 'components', 'user'), ignore_errors=True)
            shutil.move(tmp_dir + '/components/', os.path.join(path, 'components', 'user'))

    def make_sure_user_components_exists():
        for folder_name in os.listdir(os.path.join(path, 'components/standard/')):
            if not os.path.exists(path + '/components/user/' + folder_name):
                os.mkdir(path + '/components/user/' + folder_name)

    def copy_old_configs(tmp_dir):
        for config_file in ['mdt.conf', 'mdt.gui.conf']:
            if os.path.exists(tmp_dir + '/' + config_file):
                shutil.copy(tmp_dir + '/' + config_file, path + '/' + config_file)

    with tmp_save_latest_version() as tmp_dir:
        if pass_if_exists:
            if os.path.exists(path):
                return path
        else:
            if os.path.exists(path):
                shutil.rmtree(path)

        init_from_mdt()
        copy_user_components(tmp_dir)
        make_sure_user_components_exists()
        copy_old_configs(tmp_dir)

    return path


def check_user_components():
    """Check if the components in the user's home folder are up to date with this version of MDT

    Returns:
        bool: True if the .mdt folder for this version exists. False otherwise.
    """
    return os.path.isdir(get_config_dir())


def setup_logging(disable_existing_loggers=None):
    """Setup global logging.

    This uses the loaded config settings to set up the logging.

    Args:
        disable_existing_loggers (boolean): If we would like to disable the existing loggers when creating this one.
            None means use the default from the config, True and False overwrite the config.
    """
    conf = get_logging_configuration_dict()
    if disable_existing_loggers is not None:
        conf['disable_existing_loggers'] = True
    logging_config.dictConfig(conf)


def configure_per_model_logging(output_path, overwrite=False):
    """Set up logging for one specific model.

    Args:
        output_path: the output path where the model results are stored.
        overwrite (boolean): if we want to overwrite or append. If overwrite is True we overwrite the file, if False we
            append.
    """
    if output_path:
        output_path = os.path.abspath(os.path.join(output_path, 'info.log'))

    had_this_output_file = all(h.output_file == output_path for h in ModelOutputLogHandler.__instances__)

    if overwrite:
        # close any open files
        for handler in ModelOutputLogHandler.__instances__:
            handler.output_file = None
        if os.path.isfile(output_path):
            os.remove(output_path)

    for handler in ModelOutputLogHandler.__instances__:
        handler.output_file = output_path

    logger = logging.getLogger(__name__)
    if not had_this_output_file:
        if output_path:
            logger.info('Started appending to the per model log file')
        else:
            logger.info('Stopped appending to the per model log file')


@contextmanager
def per_model_logging_context(output_path, overwrite=False):
    """A logging context wrapper for the function configure_per_model_logging.

    Args:
        output_path: the output path where the model results are stored.
        overwrite (boolean): if we want to overwrite an existing file (if True), or append to it (if False)
    """
    configure_per_model_logging(output_path, overwrite=overwrite)
    yield
    configure_per_model_logging(None)


def create_sort_matrix(input_4d_vol, reversed_sort=False):
    """Create an index matrix that sorts the given input on the 4th volume from small to large values (per voxel).

    This uses

    Args:
        input_4d_vol (ndarray): the 4d input volume for which we create a sort index matrix
        reversed_sort (boolean): if True we reverse the sort and we sort from large to small.

    Returns:
        ndarray: a 4d matrix with on the 4th dimension the indices of the elements in sorted order.
    """
    sort_index = np.argsort(input_4d_vol, axis=3)

    if reversed_sort:
        return sort_index[..., ::-1]

    return sort_index


def sort_volumes_per_voxel(input_volumes, sort_matrix):
    """Sort the given volumes per voxel using the sort index in the given matrix.

    What this essentially does is to look per voxel from which map we should take the first value. Then we place that
    value in the first volume and we repeat for the next value and finally for the next voxel.

    If the length of the 4th dimension is > 1 we shift the 4th dimension to the 5th dimension and sort
    the array as if the 4th dimension values where a single value. This is useful for sorting (eigen)vector matrices.

    Args:
        input_volumes (:class:`list`): list of 4d ndarray
        sort_matrix (ndarray): 4d ndarray with for every voxel the sort index

    Returns:
        :class:`list`: the same input volumes but then with every voxel sorted according to the given sort index.
    """
    if input_volumes[0].shape[3] > 1:
        volume = np.concatenate([np.reshape(m, m.shape[0:3] + (1,) + (m.shape[3],)) for m in input_volumes], axis=3)
        grid = np.ogrid[[slice(x) for x in volume.shape]]
        sorted_volume = volume[list(grid[:-2]) + [np.reshape(sort_matrix, sort_matrix.shape + (1,))] + list(grid[-1])]
        return [sorted_volume[..., ind, :] for ind in range(len(input_volumes))]
    else:
        volume = np.concatenate([m for m in input_volumes], axis=3)
        sorted_volume = volume[list(np.ogrid[[slice(x) for x in volume.shape]][:-1])+[sort_matrix]]
        return [np.reshape(sorted_volume[..., ind], sorted_volume.shape[0:3] + (1,))
                for ind in range(len(input_volumes))]


def load_problem_data(volume_info, protocol, mask, static_maps=None, gradient_deviations=None, noise_std=None):
    """Load and create the problem data object that can be given to a model

    Args:
        volume_info (string or tuple): Either an (ndarray, img_header) tuple or the full path
            to the volume (4d signal data).
        protocol (:class:`~mdt.protocols.Protocol` or str): A protocol object with the right protocol for the
            given data, or a string object with a filename to the given file.
        mask (ndarray, str): A full path to a mask file or a 3d ndarray containing the mask
        static_maps (Dict[str, val]): the dictionary with per static map the value to use.
            The value can either be an 3d or 4d ndarray, a single number or a string. We will convert all to the
            right format.
        gradient_deviations (str or ndarray): set of gradient deviations to use. In HCP WUMINN format. Set to None to
            disable.
        noise_std (number or ndarray): either None for automatic detection,
            or a scalar, or an 3d matrix with one value per voxel.

    Returns:
        DMRIProblemData: the problem data object containing all the info needed for diffusion MRI model fitting
    """
    protocol = autodetect_protocol_loader(protocol).get_protocol()
    mask = autodetect_brain_mask_loader(mask).get_data()

    if isinstance(volume_info, string_types):
        info = load_nifti(volume_info)
        signal4d = info.get_data()
        img_header = info.get_header()
    else:
        signal4d, img_header = volume_info

    if isinstance(gradient_deviations, six.string_types):
        gradient_deviations = load_nifti(gradient_deviations).get_data()

    return DMRIProblemData(protocol, signal4d, mask, img_header, static_maps=static_maps, noise_std=noise_std,
                           gradient_deviations=gradient_deviations)


def load_brain_mask(brain_mask_fname):
    """Load the brain mask from the given file.

    Args:
        brain_mask_fname (string): The path of the brain mask to use.

    Returns:
        ndarray: The loaded brain mask data
    """
    path = nifti_filepath_resolution(brain_mask_fname)
    return load_nifti(path).get_data() > 0


def load_nifti(nifti_volume):
    """Load and return a nifti file.

    This will apply path resolution if a filename without extension is given. See the function
    :func:`~mdt.utils.nifti_filepath_resolution` for details.

    Args:
        nifti_volume (string): The filename of the volume to use.

    Returns:
        nib image proxy (from nib.use)
    """
    path = nifti_filepath_resolution(nifti_volume)
    return nib.load(path)


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


def get_cl_devices():
    """Get a list of all CL devices in the system.

    The indices of the devices can be used in the model fitting/sampling functions for 'cl_device_ind'.

    Returns:
        A list of CLEnvironments, one for each device in the system.
    """
    return CLEnvironmentFactory.smart_device_selection()


def model_output_exists(model, output_folder, append_model_name_to_path=True):
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
        append_model_name_to_path (boolean): by default we will append the name of the model to the output folder.
            This is to be consistent with the way the model fitting routine places the results in the
            <output folder>/<model_name> directories. Sometimes, however you might want to skip this appending.

    Returns:
        boolean: true if the output folder exists and contains files for all the parameters of the model.
            For a cascade model it returns true if the maps of all the models exist.
    """
    if isinstance(model, string_types):
        model = get_model(model)

    from mdt.models.cascade import DMRICascadeModelInterface
    if isinstance(model, DMRICascadeModelInterface):
        return all(model_output_exists(sub_model, output_folder, append_model_name_to_path)
                   for sub_model in model.get_model_names())

    if append_model_name_to_path:
        output_path = os.path.join(output_folder, model.name)
    else:
        output_path = output_folder

    parameter_names = model.get_optimization_output_param_names()

    if not os.path.exists(output_path):
        return False

    for parameter_name in parameter_names:
        if not glob.glob(os.path.join(output_path, parameter_name + '*')):
            return False

    return True


def split_image_path(image_path):
    """Split the path to an image into three parts, the directory, the basename and the extension.

    Args:
        image_path (str): the path to an image

    Returns:
        list of str: the path, the basename and the extension
    """
    folder = os.path.dirname(image_path)
    basename = os.path.basename(image_path)

    for extension in ['.nii.gz', '.nii']:
        if basename[-len(extension):] == extension:
            return folder, basename[0:-len(extension)], extension
    return folder, basename, ''


def calculate_information_criterions(log_likelihoods, k, n):
    """Calculate various information criterions.

    Args:
        log_likelihoods (1d np array): the array with the log likelihoods
        k (int): number of parameters
        n (int): the number of instances, protocol length

    Returns:
        dict with therein the BIC, AIC and AICc which stand for the
            Bayesian, Akaike and Akaike corrected Information Criterion
    """
    criteria = {
        'BIC': -2 * log_likelihoods + k * np.log(n),
        'AIC': -2 * log_likelihoods + k * 2}

    if n > (k + 1):
        criteria.update({'AICc': -2 * log_likelihoods + k * 2 + (2 * k * (k + 1))/(n - k - 1)})

    return criteria


class ComplexNoiseStdEstimator(object):

    def estimate(self, problem_data, **kwargs):
        """Get a noise std for the entire volume.

        Args:
            problem_data (DMRIProblemData): the problem data for which to find a noise std

        Returns:
            float or ndarray: the noise sigma of the Gaussian noise in the original complex image domain

        Raises:
            :class:`~mdt.exceptions.NoiseStdEstimationNotPossible`: if we can not estimate the
                sigma using this estimator
        """
        raise NotImplementedError()


def apply_mask(volume, mask, inplace=True):
    """Apply a mask to the given input.

    Args:
        volume (str, ndarray, list, tuple or dict): The input file path or the image itself or a list,
            tuple or dict.
        mask (str or ndarray): The filename of the mask or the mask itself
        inplace (boolean): if True we apply the mask in place on the volume image. If false we do not.

    Returns:
        Depending on the input either a singla image of the same size as the input image, or a list, tuple or dict.
        This will set for all the output images the the values to zero where the mask is zero.
    """
    from six import string_types
    from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader

    mask = autodetect_brain_mask_loader(mask).get_data()

    def apply(volume, mask):
        if isinstance(volume, string_types):
            volume = load_nifti(volume).get_data()

        mask = mask.reshape(mask.shape + (volume.ndim - mask.ndim) * (1,))

        if len(mask.shape) < 4:
            mask = mask.reshape(mask.shape + (1,))

        if len(volume.shape) < 4:
            volume = volume.reshape(volume.shape + (1,))

        if inplace:
            volume *= mask
            return volume
        return volume * mask

    if isinstance(volume, tuple):
        return (apply(v, mask) for v in volume)
    elif isinstance(volume, list):
        return [apply(v, mask) for v in volume]
    elif isinstance(volume, dict):
        return {k: apply(v, mask) for k, v in volume.items()}

    return apply(volume, mask)


def apply_mask_to_file(input_fname, mask, output_fname=None):
    """Apply a mask to the given input (nifti) file.

    If no output filename is given, the input file is overwritten.

    Args:
        input_fname (str): The input file path
        mask (str or ndarray): The mask to use
        output_fname (str): The filename for the output file (the masked input file).
    """
    mask = autodetect_brain_mask_loader(mask).get_data()

    if output_fname is None:
        output_fname = input_fname

    nib.Nifti1Image(apply_mask(input_fname, mask), None, load_nifti(input_fname).get_header()).to_filename(output_fname)


def load_samples(data_folder, mode='r'):
    """Load sampled results as a dictionary of numpy memmap.

    Args:
        data_folder (str): the folder from which to use the samples
        mode (str): the mode in which to open the memory mapped sample files (see numpy mode parameter)

    Returns:
        dict: the memory loaded samples per sampled parameter.
    """
    data_dict = {}
    for fname in glob.glob(os.path.join(data_folder, '*.samples.npy')):
        samples = open_memmap(fname, mode=mode)
        map_name = os.path.basename(fname)[0:-len('.samples.npy')]
        data_dict.update({map_name: samples})
    return data_dict


def estimate_noise_std(problem_data, estimator=None):
    """Estimate the noise standard deviation.

    Args:
        problem_data (DMRIProblemData): the problem data we can use to do the estimation
        estimator (ComplexNoiseStdEstimator): the estimator to use for the estimation. If not set we use
            the one in the configuration.

    Returns:
        the noise std estimated from the data. This can either be a single float, or an ndarray.

    Raises:
        :class:`~mdt.exceptions.NoiseStdEstimationNotPossible`: if the noise could not be estimated
    """
    logger = logging.getLogger(__name__)
    logger.info('Trying to estimate a noise std.')

    def estimate(estimation_routine):
        noise_std = estimator.estimate(problem_data)

        if isinstance(noise_std, np.ndarray) and not is_scalar(noise_std):
            logger.info('Found voxel-wise noise std using estimator {}.'.format(estimation_routine))
            return noise_std

        if np.isfinite(noise_std) and noise_std > 0:
            logger.info('Found global noise std {} using estimator {}.'.format(noise_std, estimation_routine))
            return noise_std

    if estimator:
        estimators = [estimator]
    else:
        estimators = get_noise_std_estimators()

    if len(estimators) == 1:
        return estimate(estimators[0])
    else:
        for estimator in estimators:
            try:
                return estimate(estimator)
            except NoiseStdEstimationNotPossible:
                pass

    raise NoiseStdEstimationNotPossible('Estimating the noise was not possible.')


class AutoDict(defaultdict):

    def __init__(self):
        """Create an auto-vivacious dictionary."""
        super(AutoDict, self).__init__(AutoDict)

    def to_normal_dict(self):
        """Convert this dictionary to a normal dict (recursive).

        Returns:
            dict: a normal dictionary with the items in this dictionary.
        """
        results = {}
        for key, value in self.items():
            if isinstance(value, AutoDict):
                value = value.to_normal_dict()
            results.update({key: value})
        return results


def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memmapped array values, in contrast to the numpy isscalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: true if the value is a scalar, false otherwise.
    """
    return mot.utils.is_scalar(value)


def roi_index_to_volume_index(roi_indices, brain_mask):
    """Get the 3d index of a voxel given the linear index in a ROI created with the given brain mask.

    This is the inverse function of :func:`volume_index_to_roi_index`.

    This function is useful if you, for example, have sampling results of a specific voxel
    and you want to locate that voxel in the brain maps.

    Please note that this function can be memory intensive for a large list of roi_indices

    Args:
        roi_indices (int or ndarray): the index in the ROI created by that brain mask
        brain_mask (str or 3d array): the brain mask you would like to use

    Returns:
        ndarray: the 3d voxel location(s) of the indicated voxel(s)
    """
    mask = autodetect_brain_mask_loader(brain_mask).get_data()
    return np.argwhere(mask)[roi_indices, :]


def volume_index_to_roi_index(volume_index, brain_mask):
    """Get the ROI index given the volume index (in 3d).

    This is the inverse function of :func:`roi_index_to_volume_index`.

    This function is useful if you want to locate a voxel in the ROI given the position in the volume.

    Args:
        volume_index (tuple): the volume index, a tuple or list of length 3
        brain_mask (str or 3d array): the brain mask you would like to use

    Returns:
        int: the index of the given voxel in the ROI created by the given mask
    """
    return create_index_matrix(brain_mask)[volume_index]


def create_index_matrix(brain_mask):
    """Get a matrix with on every 3d position the linear index number of that voxel.

    This function is useful if you want to locate a voxel in the ROI given the position in the volume.

    Args:
        brain_mask (str or 3d array): the brain mask you would like to use

    Returns:
        3d ndarray: a 3d volume of the same size as the given mask and with as every non-zero element the position
            of that voxel in the linear ROI list.
    """
    mask = autodetect_brain_mask_loader(brain_mask).get_data()
    roi = np.arange(0, np.count_nonzero(mask))
    return restore_volumes(roi, mask, with_volume_dim=False)


def get_temporary_results_dir(user_value):
    """Get the temporary results dir from the user value and from the config.

    Args:
        user_value (string, boolean or None): if a string is given we will use that directly. If a boolean equal to
            True is given we will use the configuration defined value. If None/False is given we will not use a specific
            temporary results dir.

    Returns:
        str or None: either the temporary results dir or None
    """
    if isinstance(user_value, string_types):
        return user_value
    if user_value is True:
        return get_tmp_results_dir()
    return None


def nifti_filepath_resolution(file_path):
    """Tries to resolve the filename to a nifti based on only the filename.

    For example, this resolves the path: ``/tmp/mask`` to:

        - ``/tmp/mask`` if exists
        - ``/tmp/mask.nii`` if exist
        - ``/tmp/mask.nii.gz`` if exists

    Hence, the lookup order is: ``path``, ``path.nii``, ``path.nii.gz``

    If a file with an extension is given we will do no further resolving and return the path as is.

    Args:
        file_path (str): the path to the nifti file, can be without extension.

    Returns:
        str: the file path we resolved to the final file.

    Raises:
        ValueError: if no nifti file could be found
    """
    if file_path[:-len('.nii')] == '.nii' or file_path[:-len('.nii.gz')] == '.nii.gz':
        return file_path

    if os.path.isfile(file_path):
        return file_path
    elif os.path.isfile(file_path + '.nii'):
        return file_path + '.nii'
    elif os.path.isfile(file_path + '.nii.gz'):
        return file_path + '.nii.gz'
    raise ValueError('No nifti file could be found using the path {}.'.format(file_path))


def create_blank_mask(volume4d_path, output_fname):
    """Create a blank mask for the given 4d volume.

    Sometimes you want to use all the voxels in the given dataset, without masking any voxel. Since the optimization
    routines require a mask, you have to submit one. The solution is to use a blank mask, that is, a mask that
    masks nothing.

    Args:
        volume4d_path (str): the path to the 4d volume you want to create a blank mask for
        output_fname (str): the path to the result mask
    """
    volume_info = load_nifti(volume4d_path)
    mask = np.ones(volume_info.shape[:3])
    nib.Nifti1Image(mask, None, volume_info.get_header()).to_filename(output_fname)


def volume_merge(volume_paths, output_fname, sort=False):
    """Merge a list of volumes on the 4th dimension. Writes the result as a file.

    You can enable sorting the list of volume names based on a natural key sort. This is
    the most convenient option in the case of globbing files. By default this behaviour is disabled.

    Example usage with globbing:

    .. code-block:: python

        mdt.volume_merge(glob.glob('*.nii'), 'merged.nii.gz', True)

    Args:
        volume_paths (list of str): the list with the input filenames
        output_fname (str): the output filename
        sort (boolean): if true we natural sort the list of DWI images before we merge them. If false we don't.
            The default is True.

    Returns:
        list of str: the list with the filenames in the order of concatenation.
    """
    images = []
    header = None

    if sort:
        def natural_key(_str):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', _str)]
        volume_paths.sort(key=natural_key)

    for volume in volume_paths:
        nib_container = load_nifti(volume)
        header = header or nib_container.get_header()
        image_data = nib_container.get_data()

        if len(image_data.shape) < 4:
            image_data = np.expand_dims(image_data, axis=3)

        images.append(image_data)

    combined_image = np.concatenate(images, axis=3)
    nib.Nifti1Image(combined_image, None, header).to_filename(output_fname)

    return volume_paths


def concatenate_mri_sets(items, output_volume_fname, output_protocol_fname, overwrite_if_exists=False):
    """Concatenate two or more DMRI datasets. Normally used to concatenate different DWI shells into one image.

    This writes a single volume and a single protocol file.

    Args:
        items (tuple of dict): A tuple of dicts with volume filenames and protocol filenames:

            .. code-block:: python

                (
                 {'volume': volume_fname,
                  'protocol': protocol_filename
                 }, ...
                )
        output_volume_fname (string): The name of the output volume
        output_protocol_fname (string): The name of the output protocol file
        overwrite_if_exists (boolean, optional, default false): Overwrite the output files if they already exists.
    """
    if not items:
        return

    if os.path.exists(output_volume_fname) and os.path.exists(output_protocol_fname) and not overwrite_if_exists:
        return

    to_concat = []
    nii_header = None

    for e in items:
        signal_img = load_nifti(e['volume'])
        signal4d = signal_img.get_data()
        nii_header = signal_img.get_header()

        protocol = load_protocol(e['protocol'])
        to_concat.append((protocol, signal4d))

    protocol, signal4d = concatenate_two_mri_measurements(to_concat)
    nib.Nifti1Image(signal4d, None, nii_header).to_filename(output_volume_fname)
    write_protocol(protocol, output_protocol_fname)


def create_median_otsu_brain_mask(dwi_info, protocol, output_fname=None, **kwargs):
    """Create a brain mask and optionally write it.

    It will always return the mask. If output_fname is set it will also write the mask.

    Args:
        dwi_info (string or tuple or image): the dwi info, either:

            - the filename of the input file;
            - or a tuple with as first index a ndarray with the DWI and as second index the header;
            - or only the image as an ndarray

        protocol (string or :class:`~mdt.protocols.Protocol`): The filename of the protocol file or a Protocol object
        output_fname (string): the filename of the output file. If None, no output is written.
            If dwi_info is only an image also no file is written.
        **kwargs: the additional arguments for the function median_otsu.

    Returns:
        ndarray: The created brain mask
    """
    from mdt.masking import create_median_otsu_brain_mask, create_write_median_otsu_brain_mask

    if output_fname:
        if not isinstance(dwi_info, (string_types, tuple, list)):
            raise ValueError('No header obtainable, can not write the brain mask.')
        return create_write_median_otsu_brain_mask(dwi_info, protocol, output_fname, **kwargs)
    return create_median_otsu_brain_mask(dwi_info, protocol, **kwargs)


def extract_volumes(input_volume_fname, input_protocol, output_volume_fname, output_protocol, volume_indices):
    """Extract volumes from the given volume and save them to separate files.

    This will index the given input volume in the 4th dimension, as is usual in multi shell DWI files.

    Args:
        input_volume_fname (str): the input volume from which to get the specific volumes
        input_protocol (str or :class:`~mdt.protocols.Protocol`): the input protocol,
            either a file or preloaded protocol object
        output_volume_fname (str): the output filename for the selected volumes
        output_protocol (str): the output protocol for the selected volumes
        volume_indices (:class:`list`): the desired indices, indexing the input_volume
    """
    input_protocol = autodetect_protocol_loader(input_protocol).get_protocol()

    new_protocol = input_protocol.get_new_protocol_with_indices(volume_indices)
    write_protocol(new_protocol, output_protocol)

    input_volume = load_nifti(input_volume_fname)
    image_data = input_volume.get_data()[..., volume_indices]
    nib.Nifti1Image(image_data, None, input_volume.get_header()).to_filename(output_volume_fname)


def recalculate_error_measures(model, problem_data, data_dir, sigma, output_dir=None, sigma_param_name=None,
                               evaluation_model=None):
    """Recalculate the information criterion maps.

    This will write the results either to the original data directory, or to the given output dir.

    Args:
        model (str or AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize
            or the name of an model we use with get_model()
        problem_data (DMRIProblemData): the problem data object
        data_dir (str): the directory containing the results for the given model
        sigma (float): the new noise sigma we use for calculating the log likelihood and then the
            information criteria's.
        output_dir (str): if given, we write the output to this directory instead of the data dir.
        sigma_param_name (str): the name of the parameter to which we will set sigma. If not given we search
            the result maps for something ending in .sigma
        evaluation_model: the evaluation model, we will manually fix the sigma in this function
    """
    from mdt.models.cascade import DMRICascadeModelInterface

    logger = logging.getLogger(__name__)

    if isinstance(model, string_types):
        model = get_model(model)

    if isinstance(model, DMRICascadeModelInterface):
        raise ValueError('This function does not accept cascade models.')

    model.set_problem_data(problem_data)

    results_maps = create_roi(Nifti.read_volume_maps(data_dir), problem_data.mask)

    if sigma_param_name is None:
        sigma_params = list(filter(lambda key: '.sigma' in key, model.get_optimization_output_param_names()))

        if not sigma_params:
            raise ValueError('Could not find a suitable parameter to set sigma for.')

        sigma_param_name = sigma_params[0]
        logger.info('Setting the given sigma value to the model parameter {}.'.format(sigma_param_name))

    model.fix(sigma_param_name, sigma)

    evaluation_model = evaluation_model or OffsetGaussianEvaluationModel()
    evaluation_model.set_noise_level_std(sigma)

    log_likelihood_calc = LogLikelihoodCalculator()
    log_likelihoods = log_likelihood_calc.calculate(model, results_maps, evaluation_model=evaluation_model)

    k = model.get_nmr_estimable_parameters()
    n = problem_data.get_nmr_inst_per_problem()
    results_maps.update({'LogLikelihood': log_likelihoods})
    results_maps.update(calculate_information_criterions(log_likelihoods, k, n))

    volumes = restore_volumes(results_maps, problem_data.mask)

    output_dir = output_dir or data_dir
    Nifti.write_volume_maps(volumes, output_dir, problem_data.volume_header)


def create_signal_estimates(volume_maps, problem_data, model, output_fname):
    """Estimate and write the signals of a given model on the given data.

    Args:
        volume_maps (str or dict): either a directory file name or a dictionary containing the results
        problem_data (DMRIProblemData): the problem data object, we will set this to the model
        model (str or model): the model or the name of the model to use for estimating the signals
        output_fname (str): the file name of the file to write the signal estimates to (.nii or .nii.gz)
    """
    if isinstance(model, string_types):
        model = get_model(model)

    if isinstance(volume_maps, string_types):
        volume_maps = Nifti.read_volume_maps(volume_maps)

    model.set_problem_data(problem_data)

    calculator = CalculateModelEstimates()
    results = calculator.calculate(model, create_roi(volume_maps, problem_data.mask))

    signal_estimates = restore_volumes(results, problem_data.mask)

    nib.Nifti1Image(signal_estimates, None, problem_data.volume_header).to_filename(output_fname)
