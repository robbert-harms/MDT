from collections import defaultdict
import collections
import copy
import distutils.dir_util
import functools
import glob
import logging
import logging.config as logging_config
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
import nibabel as nib
import numpy as np
import pickle
import pkg_resources
import six
from scipy.special import jnp_zeros
from six import string_types
import mdt.configuration as configuration
from mdt import create_index_matrix
from mdt.IO import Nifti
from mdt.cl_routines.mapping.calculate_eigenvectors import CalculateEigenvectors
from mdt.components_loader import get_model, ProcessingStrategiesLoader, NoiseSTDCalculatorsLoader
from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
from mdt.data_loaders.protocol import autodetect_protocol_loader
from mdt.data_loaders.static_maps import autodetect_static_maps_loader
from mdt.log_handlers import ModelOutputLogHandler
from mot.base import AbstractProblemData
from mot.cl_environments import CLEnvironmentFactory
from mot.cl_routines.optimizing.meta_optimizer import MetaOptimizer
from mot.factory import get_load_balance_strategy_by_name, get_optimizer_by_name, get_filter_by_name

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

    def __init__(self, protocol_data_dict, dwi_volume, mask, volume_header, static_maps=None):
        """This overrides the standard problem data to include a mask.

        Args:
            protocol_data_dict (Protocol): The protocol object used as input data to the model
            dwi_volume (ndarray): The DWI data (4d matrix)
            mask (ndarray): The mask used to create the observations list
            volume_header (nifti header): The header of the nifti file to use for writing the results.
            static_maps (Dict[str, ndarray]): the static maps used as values for the static map parameters

        Attributes:
            dwi_volume (ndarray): The DWI volume
            volume_header (nifti header): The header of the nifti file to use for writing the results.
        """
        self.dwi_volume = dwi_volume
        self.volume_header = volume_header
        self._mask = mask
        self._protocol_data_dict = protocol_data_dict
        self._observation_list = None
        self._static_maps = static_maps or {}

    @property
    def protocol(self):
        """Return the protocol_data_dict.

        Returns:
            Protocol: The protocol object given in the instantiation.
        """
        return self.protocol_data_dict

    @property
    def protocol_data_dict(self):
        """Return the constant data stored in this problem data container.

        Returns:
            dict: The protocol data dict.
        """
        return self._protocol_data_dict

    @property
    def observations(self):
        """Return the observations stored in this problem data container.

        Returns:
            ndarray: The list of observations
        """
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

    @property
    def static_maps(self):
        """Get the static maps. They are used as data for the static parameters.

        Returns:
            Dict[str, val]: per static map the value for the static map. This can either be an one or two dimensional
                matrix containing the values for each problem instance or it can be a single value we will use
                for all problem instances.
        """
        return self._static_maps

    @mask.setter
    def mask(self, new_mask):
        """Set the new mask and update the observations list.

        Args:
            new_mask (np.array): the new mask
        """
        self._mask = new_mask
        if self._observation_list is not None:
            self._observation_list = create_roi(self.dwi_volume, self._mask)


class PathJoiner(object):

    def __init__(self, *args):
        """The path joining class.

        To construct use something like:
        pjoin = PathJoiner(r'/my/images/dir/')

        or:
        pjoin = PathJoiner('my', 'images', 'dir')


        Then, you can call it like:
        pjoin()
        /my/images/dir

        At least, it returns the above on Linux. On windows it will return 'my\\images\\dir'.

        You can also call it with additional path elements which should be appended to the path:
        pjoin('/brain_mask.nii.gz')
        /my/images/dir/brain_mask.nii.gz

        Note that that is not permanent. To make it permanent you can call
        pjoin.append('results')

        This will extend the stored path to /my/images/dir/results/:
        pjoin('/brain_mask.nii.gz')
        /my/images/dir/results/brain_mask.nii.gz

        You can revert this by calling:
        pjoin.reset()

        You can also create a copy of this class with extended path elements by calling
        pjoin2 = pjoin.create_extended('results')

        This returns a new PathJoiner instance with as path the current path plus the items in the arguments.
        pjoin2('brain_mask.nii.gz')
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

    def reset(self, *args):
        """Reset the path to the path at construction time"""
        self._path = self._initial_path
        return self

    def __call__(self, *args, **kwargs):
        return os.path.abspath(os.path.join(self._path, *args))


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
        data (string, ndarray or dict): a brain volume with four dimensions (x, y, z, w)
            where w is the length of the protocol, or a list, tuple or dictionary with volumes or a string
            with a filename of a dataset to load.
        brain_mask (ndarray or str): the mask indicating the region of interest with dimensions: (x, y, z) or the string
            to the brain mask to load

    Returns:
        Signal lists for each of the given volumes. The axis are: (voxels, protocol)
    """
    from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
    brain_mask = autodetect_brain_mask_loader(brain_mask).get_data()

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
    elif isinstance(data, six.string_types):
        return creator(nib.load(data).get_data())
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
    from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
    brain_mask = autodetect_brain_mask_loader(brain_mask).get_data()

    shape3d = brain_mask.shape[:3]
    indices = np.ravel_multi_index(np.nonzero(brain_mask)[:3], shape3d, order='C')

    def restorer(voxel_list):
        s = voxel_list.shape

        def restore_3d(voxels):
            volume_length = functools.reduce(lambda x, y: x*y, shape3d)

            return_volume = np.zeros((volume_length,), dtype=voxels.dtype, order='C')
            return_volume[indices] = voxels

            return np.reshape(return_volume, shape3d)

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
        of voxels and the second dimension holds the number of evecs and the last dimension the direction per evec.

        This gives per voxel a matrix:
            [evec_1_x, evec_1_y, evec_1_z,
             evec_2_x, evec_2_y, evec_2_z
             evec_3_x, evec_3_y, evec_3_z]

        The resulting eigenvectors are the same as those from the Tensor.
    """
    return CalculateEigenvectors().convert_theta_phi_psi(theta, phi, psi)


def init_user_settings(pass_if_exists=True, keep_config=True):
    """Initializes the user settings folder using a skeleton.

    This will create all the necessary directories for adding components to MDT. It will also create a basic
    configuration file for setting global wide MDT options. Also, it will copy the user components from the previous
    version to this version.

    Each MDT version will have it's own sub-directory in the config directory.

    Args:
        pass_if_exists (boolean): if the folder for this version already exists, we might do nothing (if True)
        keep_config (boolean): if the folder for this version already exists, do we want to pass_if_exists the
            config file yes or no. This only holds for the config file.

    Returns:
        the path the user settings skeleton was written to
    """
    from mdt import get_config_dir
    path = get_config_dir()
    base_path = os.path.dirname(get_config_dir())

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    @contextmanager
    def tmp_save_previous_version():
        previous_versions = list(reversed(sorted(os.listdir(base_path))))
        tmp_dir = tempfile.mkdtemp()

        if previous_versions:
            previous_version = previous_versions[0]

            if os.path.exists(os.path.join(base_path, previous_version, 'components', 'user')):
                shutil.copytree(os.path.join(base_path, previous_version, 'components', 'user'),
                                tmp_dir + '/components/')

            if os.path.isfile(os.path.join(base_path, previous_version, 'mdt.conf')):
                shutil.copy(os.path.join(base_path, previous_version, 'mdt.conf'), tmp_dir + '/mdt.conf')

        yield tmp_dir
        shutil.rmtree(tmp_dir)

    def init_from_mdt():
        cache_path = pkg_resources.resource_filename('mdt', 'data/components')
        distutils.dir_util.copy_tree(cache_path, os.path.join(path, 'components'))

        cache_path = pkg_resources.resource_filename('mdt', 'data/mdt.conf')
        shutil.copy(cache_path, path)

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

    def copy_old_config(tmp_dir):
        if os.path.exists(tmp_dir + '/mdt.conf'):
            if os.path.exists(path + '/mdt.conf'):
                os.remove(path + '/mdt.conf')
            shutil.move(tmp_dir + '/mdt.conf', path + '/mdt.conf')

    with tmp_save_previous_version() as tmp_dir:
        if pass_if_exists:
            if os.path.exists(path):
                return path
        else:
            if os.path.exists(path):
                shutil.rmtree(path)

        init_from_mdt()
        copy_user_components(tmp_dir)
        make_sure_user_components_exists()

        if keep_config and pass_if_exists:
            copy_old_config(tmp_dir)

    return path


def check_user_components():
    """Check if the components in the user's home folder are up to date with this version of MDT

    Returns:
        bool: True if the .mdt folder for this version exists. False otherwise.
    """
    from mdt import get_config_dir
    return os.path.isdir(get_config_dir())


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
    if output_path:
        output_path = os.path.abspath(os.path.join(output_path, 'info.log'))

    had_this_output_file = all(h.output_file == output_path for h in ModelOutputLogHandler.__instances__)

    for handler in ModelOutputLogHandler.__instances__:
        handler.output_file = output_path

    logger = logging.getLogger(__name__)
    if not had_this_output_file:
        if output_path:
            logger.info('Started appending to the per model log file')
        else:
            logger.info('Stopped appending to the per model log file')


@contextmanager
def per_model_logging_context(output_path):
    """A logging context wrapper for the function configure_per_model_logging.

    Args:
        output_path: the output path where the model results are stored.
    """
    configure_per_model_logging(output_path)
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
        input_volumes (list): list of 4d ndarray
        sort_matrix (ndarray): 4d ndarray with for every voxel the sort index

    Returns:
        list: the same input volumes but then with every voxel sorted according to the given sort index.
    """
    if input_volumes[0].shape[3] > 1:
        volume = np.concatenate([np.reshape(m, m.shape[0:3] + (1,) + (m.shape[3],)) for m in input_volumes], axis=3)
        grid = np.ogrid[[slice(x) for x in volume.shape]]
        sorted_volume = volume[list(grid[:-2]) + [np.reshape(sort_matrix, sort_matrix.shape + (1,))] + list(grid[-1])]
        return [sorted_volume[..., ind, :] for ind in range(len(input_volumes))]
    else:
        volume = np.concatenate([m for m in input_volumes], axis=3)
        sorted_volume = volume[list(np.ogrid[[slice(x) for x in volume.shape]][:-1])+[sort_matrix]]
        return [np.reshape(sorted_volume[..., ind], sorted_volume.shape[0:3] + (1,)) for ind in range(len(input_volumes))]


def recursive_merge_dict(dictionary, update_dict, in_place=False):
    """ Recursively merge the given dictionary with the new values.

    If in_place is false this does not merge in place but creates new dictionary.

    If update_dict is None we return the original dictionary, or a copy if in_place is False.

    Args:
        dictionary (dict): the dictionary we want to update
        update_dict (dict): the dictionary with the new values
        in_place (boolean): if true, the changes are in place in the first dict.

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

            In the case of lists in the dictionary, we do no merging and always use the new value.
    """
    if not in_place:
        dictionary = copy.deepcopy(dictionary)

    if not update_dict:
        return dictionary

    def merge(d, upd):
        for k, v in upd.items():
            if isinstance(v, collections.Mapping):
                r = merge(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = upd[k]
        return d

    return merge(dictionary, update_dict)


def load_problem_data(volume_info, protocol, mask, static_maps=None, dtype=np.float32):
    """Load and create the problem data object that can be given to a model

    Args:
        volume_info (string): Either an (ndarray, img_header) tuple or the full path to the volume (4d signal data).
        protocol (Protocol or string): A protocol object with the right protocol for the given data,
            or a string object with a filename to the given file.
        mask (ndarray, string): A full path to a mask file or a 3d ndarray containing the mask
        static_maps (Dict[str, val]): the dictionary with per static map the value to use.
            The value can either be an 3d or 4d ndarray, a single number or a string. We will convert all to the
            right format.
        dtype (dtype) the datatype in which to load the signal volume.

    Returns:
        DMRIProblemData: the problem data object containing all the info needed for diffusion MRI model fitting
    """
    protocol = autodetect_protocol_loader(protocol).get_protocol()
    mask = autodetect_brain_mask_loader(mask).get_data()

    if isinstance(volume_info, string_types):
        signal4d, img_header = load_volume(volume_info, dtype=dtype)
    else:
        signal4d, img_header = volume_info

    if static_maps is not None:
        static_maps = {key: autodetect_static_maps_loader(val).get_data(mask) for key, val in static_maps.items()}

    return DMRIProblemData(protocol, signal4d, mask, img_header, static_maps=static_maps)


def load_volume(volume_fname, ensure_4d=True, dtype=np.float32):
    """Load the given image data from the given volume filename.

    Args:
        volume_fname (string): The filename of the volume to load.
        ensure_4d (boolean): if True we ensure that the output data matrix is in 4d.
        dtype (dtype): the numpy datatype we use for the output matrix

    Returns:
        a tuple with (data, header) for the given file.
    """
    info = nib.load(volume_fname)
    header = info.get_header()
    data = info.get_data().astype(dtype, copy=False)
    if ensure_4d:
        if len(data.shape) < 4:
            data = np.expand_dims(data, axis=3)
    return data, header


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
                The dict should only contain the elements inside optimization_settings.general
                Example config dict:
                    meta_optimizer_config = {
                        'optimizers': [{'name': 'NMSimplex', 'patience': 30, 'optimizer_options': {} }],
                        'extra_optim_runs': 0,
                        ...
                    }
        """
        self._meta_optimizer_config = meta_optimizer_config or {}

    def construct(self, model_names=None):
        """Construct a new meta optimizer with the options from the current configuration.

        If model_name is given, we try to load the specific options for that model from the configuration. If it it not
        given we load the general options under 'general/meta_optimizer'.

        Args:
            model_names (list of str): the list of model names
        """
        optim_config = self._get_configuration_dict(model_names)

        cl_environments = self._get_cl_environments(optim_config)
        load_balancer = self._get_load_balancer(optim_config)

        meta_optimizer = MetaOptimizer(cl_environments, load_balancer)

        meta_optimizer.optimizer = self._get_optimizer(optim_config['optimizers'][0], cl_environments, load_balancer)
        meta_optimizer.extra_optim_runs_optimizers = [self._get_optimizer(optim_config['optimizers'][i],
                                                                          cl_environments, load_balancer)
                                                      for i in range(1, len(optim_config['optimizers']))]

        for attr in ('extra_optim_runs', 'extra_optim_runs_apply_smoothing', 'extra_optim_runs_use_perturbation'):
            meta_optimizer.__setattr__(attr, optim_config[attr])

        if 'smoothing_routines' in optim_config and len(optim_config['smoothing_routines']):
            meta_optimizer.smoother = self._get_smoother(optim_config['smoothing_routines'][0],
                                                         cl_environments, load_balancer)
            meta_optimizer.extra_optim_runs_smoothers = [self._get_smoother(optim_config['smoothing_routines'][i],
                                                                            cl_environments, load_balancer)
                                                         for i in range(1, len(optim_config['smoothing_routines']))]

        return meta_optimizer

    def _get_load_balancer(self, optim_config):
        load_balancer = get_load_balance_strategy_by_name(optim_config['load_balancer']['name'])()
        for attr, value in optim_config['load_balancer'].items():
            if attr != 'name':
                load_balancer.__setattr__(attr, value)
        return load_balancer

    def _get_cl_environments(self, optim_config):
        cl_environments = CLEnvironmentFactory.all_devices()
        if optim_config['cl_devices']:
            if isinstance(optim_config['cl_devices'], (tuple, list)):
                cl_environments = [cl_environments[int(ind)] for ind in optim_config['cl_devices']]
            else:
                cl_environments = [cl_environments[int(optim_config['cl_devices'])]]
        return cl_environments

    def _get_configuration_dict(self, model_names):
        current_config = configuration.config['optimization_settings']
        optim_config = current_config['general']

        if model_names and 'model_specific' in current_config:
            info_dict = get_model_config(model_names, current_config['model_specific'])
            if info_dict:
                optim_config = recursive_merge_dict(optim_config, info_dict)

        optim_config = recursive_merge_dict(optim_config, self._meta_optimizer_config)
        return optim_config

    def _get_optimizer(self, options, cl_environments, load_balancer):
        optimizer = get_optimizer_by_name(options['name'])
        patience = options['patience']
        optimizer_options = options.get('optimizer_options')
        return optimizer(cl_environments, load_balancer, patience=patience, optimizer_options=optimizer_options)

    def _get_smoother(self, options, cl_environments, load_balancer):
        smoother = get_filter_by_name(options['name'])
        size = options['size']
        return smoother(size, cl_environments, load_balancer)


def get_cl_devices():
    """Get a list of all CL devices in the system.

    The indices of the devices can be used in the model fitting/sampling functions for 'cl_device_ind'.

    Returns:
        A list of CLEnvironments, one for each device in the system.
    """
    return CLEnvironmentFactory.all_devices()


def get_model_config(model_names, config):
    """Get from the given dictionary the config for the given model.

    This tries to find the best match between the given config items (by key) and the given model list. For example
    if model_names is ['BallStick', 'S0'] and we have the following config dict:
        {'^S0$': 0,
         '^BallStick$': 1
         ('^BallStick$', '^S0$'): 2,
         ('^BallStickStick$', '^BallStick$', '^S0$'): 3,
         }

    then this function should return 2. because that one is the best match, even though the last option is also a
    viable match. That is, since a subset of the keys in the last option also matches the model names, it is
    considered a match as well. Still the best match is the third option (returning 2).

    Args:
        model_names (list of str): the names of the models we want to match. This should contain the entire
            recursive list of cascades leading to the single model we want to get the config for.
        config (dict): the config items with as keys either a single model regex for a name or a list of regex for
            a chain of model names.

    Returns:
        The config content of the best matching key.
    """
    if not config:
        return {}

    def get_key_length(key):
        if isinstance(key, tuple):
            return len(key)
        return 1

    def is_match(model_names, config_key):
        if isinstance(model_names, string_types):
            model_names = [model_names]

        if len(model_names) != get_key_length(config_key):
            return False

        if isinstance(config_key, tuple):
            return all([re.match(config_key[ind], model_names[ind]) for ind in range(len(config_key))])

        return re.match(config_key, model_names[0])

    key_length_lookup = ((get_key_length(key), key) for key in config.keys())
    ascending_keys = tuple(item[1] for item in sorted(key_length_lookup, key=lambda info: info[0]))

    # direct matching
    for key in ascending_keys:
        if is_match(model_names, key):
            return config[key]

    # partial matching string keys to last model name
    for key in ascending_keys:
        if not isinstance(key, tuple):
            if is_match(model_names[-1], key):
                return config[key]

    # partial matching tuple keys with a moving filter
    for key in ascending_keys:
        if isinstance(key, tuple):
            for start_ind in range(len(key)):
                sub_key = key[start_ind:]

                if is_match(model_names, sub_key):
                    return config[key]

    # no match found
    return {}


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
    logger = logging.getLogger(__name__)

    if model_protocol_options:
        protocol = problem_data.protocol
        protocol_indices = np.array([], dtype=np.int64)

        if model_protocol_options.get('use_weighted', False):
            if 'b_value' in model_protocol_options:
                options = {'start': 0, 'end': 1.5e9}
                for key, value in model_protocol_options['b_value'].items():
                    options.update({key: value})
                protocol_indices = protocol.get_indices_bval_in_range(**options)

        if model_protocol_options.get('use_unweighted', False):
            unweighted_threshold = model_protocol_options.get('unweighted_threshold', None)
            protocol_indices = np.append(protocol_indices, protocol.get_unweighted_indices(unweighted_threshold))

        protocol_indices = np.unique(protocol_indices)

        if len(protocol_indices) != protocol.length:
            logger.info('Applying model protocol options, we will use a subset of the protocol and DWI.')
            logger.info('Using {} out of {} volumes, indices: {}'.format(
                len(protocol_indices), protocol.length, str(protocol_indices).replace('\n', '').replace('[  ', '[')))

            new_protocol = protocol.get_new_protocol_with_indices(protocol_indices)

            new_dwi_volume = problem_data.dwi_volume[..., protocol_indices]

            return DMRIProblemData(new_protocol, new_dwi_volume, problem_data.mask,
                                   problem_data.volume_header)
        else:
            logger.info('No model protocol options to apply, using original protocol.')

    return problem_data


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

    extension = ''
    if '.nii.gz' in basename:
        extension = '.nii.gz'
    elif '.nii' in basename:
        extension = '.nii'

    basename = basename.replace(extension, '')
    return folder, basename, extension


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

    def __init__(self, problem_data):
        """Estimation routine for estimating the standard deviation of the Gaussian error in the complex signal images.

        Args:
            problem_data the full set of problem data
        """
        self._problem_data = problem_data
        self._logger = logging.getLogger(__name__)

    def estimate(self, **kwargs):
        """Get a noise std for the entire volume.

        Returns:
            float or ndarray: the noise sigma of the Gaussian noise in the original complex image domain

        Raises:
            NoiseStdEstimationNotPossible: if we can not estimate the sigma using this estimator
        """


class NoiseStdEstimationNotPossible(Exception):
    """An exception that can be raised by any ComplexNoiseStdEstimator.

    This indicates that the noise std can not be estimated.
    """


def apply_mask(volume, mask, inplace=True):
    """Apply a mask to the given input.

    Args:
        volume (str, ndarray, list, tuple or dict): The input file path or the image itself or a list, tuple or
            dict.
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
            volume = load_volume(volume)[0]
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


class ModelProcessingStrategy(object):

    def __init__(self):
        """Model processing strategies define in what parts the model is analyzed.

        This uses the problems_to_analyze attribute of the MOT model builder to select the voxels to process. That
        attribute arranges that only a selection of the problems are analyzed instead of all of them.
        """
        self._logger = logging.getLogger(__name__)

    def run(self, model, problem_data, output_path, recalculate, worker):
        """Process the given dataset using the logistics the subclass.

        Subclasses of this base class can implement all kind of logic to divide a large dataset in smaller chunks
        (for example slice by slice) and run the processing on each slice separately and join the results afterwards.

        Args:
             model (AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize.
             problem_data (DMRIProblemData): The problem data object with which the model is initialized before running
             output_path (string): The full path to the folder where to place the output
             recalculate (boolean): If we want to recalculate the results if they are already present.
             worker (ModelProcessingWorker): the worker we use to do the processing

        Returns:
            dict: the results as a dictionary of roi lists
        """


class ModelChunksProcessingStrategy(ModelProcessingStrategy):

    def __init__(self, honor_voxels_to_analyze=True):
        """This class is a baseclass for all model slice fitting strategies that fit the data in chunks/parts.

        Args:
            honor_voxels_to_analyze (bool): if set to True, we use the model's voxels_to_analyze setting if set
                instead of fitting all voxels in the mask
        """
        super(ModelChunksProcessingStrategy, self).__init__()
        self.honor_voxels_to_analyze = honor_voxels_to_analyze

    def _prepare_chunk_dir(self, chunks_dir, recalculate):
        """Prepare the directory for a new chunk.

        Args:
            chunks_dir (str): the full path to the chunks directory.
            recalculate (boolean): if true and the data exists, we throw everything away to start over.
        """
        if recalculate:
            if os.path.exists(chunks_dir):
                shutil.rmtree(chunks_dir)

        if not os.path.exists(chunks_dir):
            os.makedirs(chunks_dir)

    @contextmanager
    def _selected_indices(self, model, chunk_indices):
        """Create a context in which problems_to_analyze attribute of the models is set to the selected indices.

        Args:
            model: the model to which to set the problems_to_analyze
            chunk_indices (ndarray): the list of voxel indices we want to use for processing
        """
        old_setting = model.problems_to_analyze
        model.problems_to_analyze = chunk_indices
        yield
        model.problems_to_analyze = old_setting


class ModelProcessingWorker(object):

    def process(self, model, problem_data, mask, output_dir):
        """Process the indicated voxels in the way prescribed by this worker.

        Since the processing strategy can use all voxels to do the analysis in one go, this function
        should return all the output it can, i.e. the same kind of output as from the function combine

        Args:
            model (DMRISingleModel): the model to process
            problem_data (DMRIProblemData): The problem data object with which the model is initialized before running
            mask (ndarray): the mask that was used in this processing step
            output_dir (str): the location for the output files

        Returns:
            the results for this single processing step
        """

    def output_exists(self, model, problem_data, output_dir):
        """Check if in the given directory all the output exists for the given model.

        This could be used by the processing strategy to check if in a given folder for a single slice all the
        output items exist.

        Args:
            model (DMRISingleModel): the model to process
            problem_data (DMRIProblemData): The problem data object with which the model is initialized before running
            output_dir (str): the location for the output files

        Returns:
            boolean: true if the output exists, false otherwise
        """

    def combine(self, model, problem_data, output_path, chunks_dir):
        """Combine all the calculated parts.

        This function should combine all the results into one compilation.

        This expects there to be a file named '__mask.nii.gz' in every sub dir. This file should contain
        a mask for exactly that slice/data calculated in that directory. We will use those chunk masks to combine
        the data for the whole dataset.

        Args:
            model (DMRISingleModel): the model we processed
            problem_data (DMRIProblemData): The problem data object with which the model is initialized before running
            output_path (str): the location for the final combined output files
            chunks_dir (str): the location of the directory that contains all the directories with the chunks.

        Returns:
            the processing results for as much as this is applicable
        """


class FittingProcessingWorker(ModelProcessingWorker):

    def __init__(self, optimizer):
        """The processing worker for model fitting.

        Use this if you want to use the model processing strategy to do model fitting.

        Args:
            optimizer: the optimization routine to use
        """
        self._optimizer = optimizer

    def process(self, model, problem_data, mask, output_dir):
        results, extra_output = self._optimizer.minimize(model, full_output=True)
        results.update(extra_output)

        self._write_output(results, mask, output_dir, problem_data.volume_header)

        return results

    def output_exists(self, model, problem_data, output_dir):
        return model_output_exists(model, output_dir, append_model_name_to_path=False)

    def combine(self, model, problem_data, output_path, chunks_dir):
        sub_dirs = list(os.listdir(chunks_dir))
        if sub_dirs:
            file_paths = glob.glob(os.path.join(chunks_dir, os.listdir(chunks_dir)[0], '*.nii*'))
            file_paths = filter(lambda d: '__mask' not in d, file_paths)
            map_names = map(lambda d: split_image_path(d)[1], file_paths)

            results = {}
            for map_name in map_names:
                map_paths = []
                mask_paths = []

                for sub_dir in sub_dirs:
                    map_file = os.path.join(chunks_dir, sub_dir, map_name + '.nii.gz')

                    if os.path.exists(map_file):
                        map_paths.append(map_file)
                        mask_paths.append(os.path.join(chunks_dir, sub_dir, '__mask.nii.gz'))

                results.update({map_name: join_parameter_maps(output_path, map_paths, mask_paths, map_name)})

            return results

    def _write_output(self, result_arrays, mask, output_path, volume_header):
        """Write the result arrays to the given output folder"""
        volume_maps = restore_volumes(result_arrays, mask)
        Nifti.write_volume_maps(volume_maps, output_path, volume_header)


class SamplingProcessingWorker(ModelProcessingWorker):

    def __init__(self, sampler):
        """The processing worker for model sampling.

        Use this if you want to use the model processing strategy to do model sampling.

        Args:
            sampler (AbstractSampler): the optimization sampler to use
        """
        self._sampler = sampler

    def process(self, model, problem_data, mask, output_dir):
        results, other_output = self._sampler.sample(model, full_output=True)
        write_sample_results(results, output_dir)
        self._write_maps(other_output, mask, problem_data, output_dir)
        return memory_load_samples(output_dir)

    def output_exists(self, model, problem_data, output_dir):
        return model_output_exists(model, os.path.join(output_dir, 'volume_maps'), append_model_name_to_path=False)

    def combine(self, model, problem_data, output_path, chunks_dir):
        self._combine_maps(output_path, chunks_dir)
        self._combine_samples(problem_data.mask, output_path, chunks_dir)
        return memory_load_samples(output_path)

    def _combine_maps(self, output_path, chunks_dir):
        sub_dirs = list(os.listdir(chunks_dir))
        if sub_dirs:
            file_paths = glob.glob(os.path.join(chunks_dir, os.listdir(chunks_dir)[0], 'volume_maps', '*.nii*'))
            file_paths = filter(lambda d: '__mask' not in d, file_paths)
            map_names = map(lambda d: split_image_path(d)[1], file_paths)

            for map_name in map_names:
                map_paths = list(os.path.join(chunks_dir, sub_dir, 'volume_maps', map_name + '.nii.gz')
                                 for sub_dir in sub_dirs)
                mask_paths = list(os.path.join(chunks_dir, sub_dir, '__mask.nii.gz') for sub_dir in sub_dirs)

                join_parameter_maps(os.path.join(output_path, 'volume_maps'), map_paths, mask_paths, map_name)

    def _combine_samples(self, whole_mask, output_path, chunks_dir):
        sub_dirs = list(os.listdir(chunks_dir))
        if sub_dirs:
            file_paths = glob.glob(os.path.join(chunks_dir, os.listdir(chunks_dir)[0], '*.samples'))
            map_names = map(lambda d: os.path.splitext(os.path.basename(d))[0], file_paths)

            for map_name in map_names:
                map_paths = list(os.path.join(chunks_dir, sub_dir, map_name + '.samples') for sub_dir in sub_dirs)
                map_settings = list(os.path.join(chunks_dir, sub_dir, map_name + '.samples.settings')
                                    for sub_dir in sub_dirs)
                mask_paths = list(os.path.join(chunks_dir, sub_dir, '__mask.nii.gz') for sub_dir in sub_dirs)

                self._join_sample_results(whole_mask, output_path, map_paths, map_settings, mask_paths, map_name)

    def _write_maps(self, other_output, mask, problem_data, output_path):
        volume_maps_dir = os.path.join(output_path, 'volume_maps')
        volume_maps = restore_volumes(other_output, mask)
        Nifti.write_volume_maps(volume_maps, volume_maps_dir, problem_data.volume_header)

    def _join_sample_results(self, whole_mask, output_path, maps, map_settings, masks, map_name):
        """Join the sample results of a single parameter.

        This uses the given masks to determine where to place which voxels.

        Args:
            whole_mask (ndarray): the complete brain mask
            output_path (str): where to place the output file
            maps (list of str): the list of sample files to concatenate
            map_settings (list of str): a list with the same length as maps containing the settings for the given
                maps
            masks (list of str): the list of masks indicating the voxels present in the map.
            Should have same length as maps.
            map_name (str): the name of the output file
        """
        settings = []

        for map_setting_file in map_settings:
            with open(map_setting_file, 'rb') as f:
                settings.append(pickle.load(f))

        total_shape = (sum(s['shape'][0] for s in settings), settings[0]['shape'][1])
        total = np.memmap(os.path.join(output_path, map_name + '.samples'), dtype=settings[0]['dtype'],
                          shape=total_shape, mode='w+')

        settings_total = {'dtype': settings[0]['dtype'], 'shape': total_shape}
        with open(os.path.join(output_path, map_name + '.samples.settings'), 'wb') as f:
            pickle.dump(settings_total, f, protocol=pickle.HIGHEST_PROTOCOL)

        whole_mask_indices = create_index_matrix(whole_mask)

        for map_fname, settings, mask_fname in zip(maps, settings, masks):
            chunk_mask = nib.load(mask_fname).get_data().astype(np.bool)
            chunk_indices = np.squeeze(create_roi(whole_mask_indices * chunk_mask, chunk_mask))

            current = np.memmap(map_fname, dtype=settings['dtype'], mode='r', shape=settings['shape'])
            total[chunk_indices, :] = current
            del current
        del total


def write_sample_results(results, output_path):
    """Write the sample results to file.

    This will write to files per sampled parameter. The first is a numpy array written to file, the second
    is a python pickled dictionary with the datatype and shape of the written numpy array.

    Args:
        results (dict): the samples to write
        output_path (str): the path to write the samples in
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for map_name, samples in results.items():
        saved = np.memmap(os.path.join(output_path, map_name + '.samples'),
                          dtype=samples.dtype, mode='w+', shape=samples.shape)
        saved[:] = samples[:]
        del saved

        settings = {'dtype': samples.dtype, 'shape': samples.shape}
        with open(os.path.join(output_path, map_name + '.samples.settings'), 'wb') as f:
            pickle.dump(settings, f, protocol=pickle.HIGHEST_PROTOCOL)


def memory_load_samples(data_folder):
    """Load sampled results as a dictionary of numpy memmap.

    Args:
        data_folder (str): the folder from which to load the samples

    Returns:
        dict: the memory loaded samples per sampled parameter.
    """
    data_dict = {}
    for fname in glob.glob(os.path.join(data_folder, '*.samples')):
        if os.path.isfile(fname + '.settings'):
            with open(fname + '.settings', 'rb') as f:
                settings = pickle.load(f)
                samples = np.memmap(fname, dtype=settings['dtype'], mode='r', shape=settings['shape'])

                map_name = os.path.splitext(os.path.basename(fname))[0]
                data_dict.update({map_name: samples})
    return data_dict


def join_parameter_maps(output_dir, maps, masks, map_name):
    """Join the chunks of a single parameter over all chunk dirs.

    This uses the given masks to determine where to place which voxels.

    Args:
        output_dir (str): where to place the concatenated output
        maps (list of str): the list of map filenames we want to concatenate.
            Should have same length as masks.
        masks (list of str): the list of masks indicating the voxels present in the map.
            Should have same length as maps.
        map_name (str): the name of the output map

    Returns:
        np.array: the values of the concatenated map in a large array
    """
    results = None
    mask_so_far = None
    volume_header = None

    for map_fname, mask_fname in zip(maps, masks):
        sub_results = nib.load(map_fname).get_data()
        mask_nib = nib.load(mask_fname)
        mask = mask_nib.get_data().astype(np.bool)

        if volume_header is None:
            volume_header = mask_nib.get_header()

        if results is None:
            results = sub_results
            mask_so_far = mask
        else:
            mask_so_far += mask
            sub_results = apply_mask(sub_results, mask_so_far)
            results += sub_results

    Nifti.write_volume_maps({map_name: results}, output_dir, volume_header)
    return create_roi(results, mask_so_far)


def get_processing_strategy(processing_type, model_names=None):
    """Get from the config file the correct processing strategy for the given model.

    Args:
        processing_type (str): 'optimization', 'sampling' or any other of the
            processing_strategies defined in the config
        model_names (list of str): the list of model names (the full recursive cascade of model names)

    Returns:
        ModelProcessingStrategy: the processing strategy to use for this model
    """
    strategy_name = configuration.config['processing_strategies'][processing_type]['general']['name']
    options = configuration.config['processing_strategies'][processing_type]['general'].get('options', {}) or {}

    if model_names and 'model_specific' in configuration.config['processing_strategies'][processing_type]:
        info_dict = get_model_config(
            model_names, configuration.config['processing_strategies'][processing_type]['model_specific'])

        if info_dict:
            strategy_name = info_dict['name']
            options = info_dict.get('options', {}) or {}

    return ProcessingStrategiesLoader().load(strategy_name, **options)


def estimate_noise_std(problem_data, estimation_cls_name=None):
    """Estimate the noise standard deviation.

    Args:
        problem_data (DMRIProblemData): the problem data we can use to do the estimation
        estimation_cls_name (str): the name of the estimation class to load. If none given we try each defined in the
            current config.

    Returns:
        the noise std estimated from the data. This can either be a single float, or an ndarray.

    Raises:
        NoiseStdEstimationNotPossible: if the noise could not be estimated
    """
    loader = NoiseSTDCalculatorsLoader()
    logger = logging.getLogger(__name__)

    def estimate(estimator_name):
        estimator = loader.get_class(estimator_name)(problem_data)
        noise_std = estimator.estimate()

        if isinstance(noise_std, np.ndarray):
            logger.info('Found voxel-wise noise std using estimator {}.'.format(noise_std, estimator_name))
            return noise_std

        if np.isfinite(noise_std) and noise_std > 0:
            logger.info('Found global noise std {} using estimator {}.'.format(noise_std, estimator_name))
            return noise_std

    if estimation_cls_name:
        estimators = [estimation_cls_name]
    else:
        estimators = configuration.config['noise_std_estimating']['general']['estimators']

    if len(estimators) == 1:
        return estimate(estimators[0])
    else:
        for estimator in estimators:
            try:
                return estimate(estimator)
            except NoiseStdEstimationNotPossible:
                pass

    raise NoiseStdEstimationNotPossible('Estimating the noise was not possible.')


def get_noise_std_value(noise_std, problem_data):
    """Convert the variable valued noise std to an actual noise std.

    This is meant to be used by the fitting and sampling routines to return a proper value for the
    noise std given the user provided noise std and the problem data.

    Args:
        noise_std (None, double, ndarray, 'auto', 'auto-local' or 'auto-global'): the noise level standard deviation.
            The different values can be:
                None: set to 1
                double: use a single value for all voxels
                ndarray: use a value per voxel. If given this should be a value for the entire dataset.
                'auto': tries to estimate the noise std from the data

        problem_data: if we need to estimate the noise std, the problem data to use

    Returns:
        either a single value or a ndarray with a value for every voxel in the volume
    """
    if noise_std is None:
        return 1
    elif isinstance(noise_std, np.ndarray):
        return noise_std

    elif noise_std == 'auto':
        logger = logging.getLogger(__name__)
        logger.info('The noise std was set to \'auto\', we will now estimate a noise std.')
        try:
            return estimate_noise_std(problem_data)
        except NoiseStdEstimationNotPossible:
            logger.warn('Failed to estimate a noise std for this subject. We will continue with an std of 1.')
            return 1

    return noise_std


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
