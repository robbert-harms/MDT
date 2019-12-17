import collections
import gzip
import hashlib
import glob
import logging
import logging.config as logging_config
import os
import re
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import pkg_resources
from numpy.lib.format import open_memmap
import mot.lib.utils
from mdt.lib.components import get_model
from mdt.configuration import get_config_dir
from mdt.configuration import get_logging_configuration_dict, get_tmp_results_dir
from mdt.lib.deferred_mappings import DeferredActionDict, DeferredActionTuple
from mdt.lib.exceptions import NoiseStdEstimationNotPossible
from mdt.lib.log_handlers import ModelOutputLogHandler
from mdt.lib.nifti import load_nifti, write_nifti
from mdt.protocols import load_protocol, write_protocol
from mot import minimize
from mot.lib.cl_environments import CLEnvironmentFactory
from mdt.model_building.parameter_functions.dependencies import AbstractParameterDependency

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Struct, Array
from mot.library_functions import dawson

DEFAULT_INTERMEDIATE_RESULTS_SUBDIR_NAME = 'tmp_results'


class InitializationData:

    def apply_to_model(self, model, input_data):
        """Apply all information in this initialization data to the given model.

        This applies the information in this init data to given model in place.

        Args:
            model: the model to apply the initializations on
            input_data (mdt.lib.input_data.MRIInputData): the input data used in the fit
        """
        raise NotImplementedError()

    def get_inits(self):
        """Get the initialization values.

        Returns:
            dict: the initialization values with per map either a scalar or a 3d/4d volume
        """
        raise NotImplementedError()

    def get_fixes(self):
        """Determines which parameters need to be fixed and to which values.

        Returns:
            dict: the initialization values with per map either a scalar or a 3d/4d volume
        """
        raise NotImplementedError()

    def get_lower_bounds(self):
        """Get the lower bounds to use in the model processing.

        Returns:
            dict: the lower bounds values with per map either a scalar or a 3d/4d volume
        """
        raise NotImplementedError()

    def get_upper_bounds(self):
        """Get the upper bounds to use in the model processing.

        Returns:
            dict: the upper bounds values with per map either a scalar or a 3d/4d volume
        """
        raise NotImplementedError()


class SimpleInitializationData(InitializationData):

    def __init__(self, inits=None, fixes=None, lower_bounds=None, upper_bounds=None, unfix=None):
        """A storage class for initialization data during model fitting and sample.

        Every element is supposed to be a dictionary with as keys the name of a parameter and as value a scalar value
        or a 3d/4d volume.

        Args:
            inits (dict): indicating the initialization values for the parameters. Example of use:

                .. code-block:: python

                    inits = {'Stick.theta': np.pi,
                             'Stick.phi': './my_init_map.nii.gz'}

            fixes (dict): indicating fixations of a parameter. Example of use:

                .. code-block:: python

                    fixes = {'Ball.d': 3.0e-9}

                As values it accepts scalars and maps but also strings defining dependencies.

            lower_bounds (dict): the lower bounds per parameter
            upper_bounds (dict): the upper bounds per parameter
            unfix (list or tuple): the list of parameters to unfix
        """
        self._inits = inits or {}
        self._fixes = fixes or {}
        self._lower_bounds = lower_bounds or {}
        self._upper_bounds = upper_bounds or {}
        self._unfix = unfix or []

    def apply_to_model(self, model, input_data):
        def prepare_value(_, v):
            if is_scalar(v):
                return v

            if isinstance(v, str):
                return v

            if isinstance(v, AbstractParameterDependency):
                return v

            return create_roi(v, input_data.mask)

        if len(self._inits):
            model.set_initial_parameters(DeferredActionDict(prepare_value, self.get_inits()))

        if len(self._lower_bounds):
            model.set_lower_bounds(DeferredActionDict(prepare_value, self.get_lower_bounds()))

        if len(self._upper_bounds):
            model.set_upper_bounds(DeferredActionDict(prepare_value, self.get_upper_bounds()))

        if len(self._fixes):
            for key, value in self.get_fixes().items():
                model.fix(key, prepare_value(key, value))

        if len(self._unfix):
            for param_name in self._unfix:
                model.unfix(param_name)

    def get_inits(self):
        return DeferredActionDict(self._resolve_value, self._inits)

    def get_fixes(self):
        return self._fixes

    def get_lower_bounds(self):
        return DeferredActionDict(self._resolve_value, self._lower_bounds)

    def get_upper_bounds(self):
        return DeferredActionDict(self._resolve_value, self._upper_bounds)

    def _resolve_value(self, key, value):
        if isinstance(value, str):
            return load_nifti(value).get_data()
        return value


class PathJoiner:

    def __init__(self, *args, make_dirs=False):
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
            make_dirs (boolean): make_dirs (boolean): if set to True we will automatically create the directory
                this path is pointing to. Similar to calling :meth:`make_dirs` on the resulting object.
        """
        self._initial_path = os.path.abspath(os.path.join('', *args))
        self._path = os.path.abspath(os.path.join('', *args))
        if make_dirs:
            self.make_dirs()

    def create_extended(self, *args, make_dirs=False, make_dirs_mode=None):
        """Create and return a new PathJoiner instance with the path extended by the given arguments.

        Args:
            make_dirs (boolean): if set to True we will automatically create the directory this path is pointing to.
                Similar to calling :meth:`make_dirs` on the resulting object.
            make_dirs_mode (int): the mode for the call to :meth:`make_dirs`.
        """
        pj = PathJoiner(os.path.join(self._path, *args))
        if make_dirs:
            pj.make_dirs(mode=make_dirs_mode)
        return pj

    def append(self, *args):
        """Extend the stored path with the given elements"""
        self._path = os.path.join(self._path, *args)
        return self

    def reset(self):
        """Reset the path to the path at construction time"""
        self._path = self._initial_path
        return self

    def make_dirs(self, dir=None, mode=None):
        """Create the directories if they do not exists.

        This first creates the directory mentioned in the path joiner. Afterwards, it will create the additional
        specified directory.

        This uses os.makedirs to make the directories. The given argument mode is handed to os.makedirs.

        Args:
            dir (str or list or str): single additional directory to create, can be a nested directory.
            mode (int): the mode parameter for os.makedirs, defaults to 0o777
        """
        if mode is None:
            mode = 0o777
        if not os.path.exists(self._path):
            os.makedirs(self._path, mode)

        if dir:
            if isinstance(dir, str):
                self.create_extended(dir, make_dirs=True, make_dirs_mode=mode)
            else:
                self.create_extended(*dir, make_dirs=True, make_dirs_mode=mode)

    def __call__(self, *args):
        if len(args) and args[0].startswith('/'):
            args = list(args)
            args[0] = args[0][1:]
        return os.path.abspath(os.path.join(self._path, *args))


def split_dataset(dataset, split_dimension, split_index):
    """Split the given dataset along the given dimension on the given index.

    Args:
        dataset (ndarray, list, tuple, dict, string): The single volume or list of volumes to split in two
        split_dimension (int): The dimension along which to split the dataset
        split_index (int): The index on the given dimension to split the volume(s)

    Returns:
        ndarray, list, tuple, dict: If dataset is a single volume return a tuple of two volumes which concatenated
            give the original volume back. If it is a list, tuple or dict we return a tuple containing two lists, tuples
            or dicts, with the same indices and with each holding half of the splitted data.
    """
    if isinstance(dataset, str):
        return split_dataset(load_nifti(dataset).get_data(), split_dimension, split_index)

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
    ind_1[split_dimension] = range(0, int(split_index))

    ind_2 = [slice(None)] * dataset.ndim
    ind_2[split_dimension] = range(int(split_index), int(dataset.shape[split_dimension]))

    return dataset[ind_1], dataset[ind_2]


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

    data_slice = get_slice_in_dimension(brain_mask, roi_dimension, roi_slice)

    roi_mask[tuple(ind_pos)] = data_slice

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

    brain_mask_img = load_nifti(brain_mask_fname)
    brain_mask = brain_mask_img.get_data()
    img_header = brain_mask_img.header
    roi_mask = create_slice_roi(brain_mask, roi_dimension, roi_slice)
    write_nifti(roi_mask, output_fname, img_header)
    return roi_mask


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
    return volume[tuple(ind_pos)]


def create_roi(data, brain_mask):
    """Create and return masked data of the given brain volume and mask

    Args:
        data (string, ndarray or dict): a brain volume with four dimensions (x, y, z, w)
            where w is the length of the protocol, or a list, tuple or dictionary with volumes or a string
            with a filename of a dataset to use or a directory with the containing maps to load.
        brain_mask (ndarray or str): the mask indicating the region of interest with dimensions: (x, y, z) or the string
            to the brain mask to use

    Returns:
        ndarray, tuple, dict: If a single ndarray is given we will return the ROI for that array. If
            an iterable is given we will return a tuple. If a dict is given we return a dict.
            For each result the axis are: (voxels, protocol)
    """
    brain_mask = load_brain_mask(brain_mask)

    def creator(v):
        return_val = v[brain_mask]
        if len(return_val.shape) == 1:
            return_val = np.expand_dims(return_val, axis=1)
        return return_val

    if isinstance(data, (dict, collections.Mapping)):
        return DeferredActionDict(lambda _, item: create_roi(item, brain_mask), data)
    elif isinstance(data, str):
        if os.path.isdir(data):
            return create_roi(load_volume_maps(data), brain_mask)
        return creator(load_nifti(data).get_data())
    elif isinstance(data, (list, tuple, collections.Sequence)):
        return DeferredActionTuple(lambda _, item: create_roi(item, brain_mask), data)
    return creator(data)


def restore_volumes(data, brain_mask, with_volume_dim=True):
    """Restore the given data to a whole brain volume

    The data can be a list, tuple or dictionary with two dimensional arrays, or a 2d array itself.

    Args:
        data (ndarray): the data as a x dimensional list of voxels, or, a list, tuple, or dict of those voxel lists
        brain_mask (ndarray): the brain_mask which was used to generate the data list
        with_volume_dim (boolean): If true we always return values with at least 4 dimensions.
            The extra dimension is for the volume index. If false we return at least 3 dimensions.

    Returns:
        Either a single whole volume, a list, tuple or dict of whole volumes, depending on the given data.
        If with_volume_ind_dim is set we return values with 4 dimensions. (x, y, z, 1). If not set we return only
        three dimensions.
    """
    brain_mask = load_brain_mask(brain_mask)

    shape3d = brain_mask.shape[:3]
    indices = np.ravel_multi_index(np.nonzero(brain_mask)[:3], shape3d, order='C')

    def restorer(voxel_list):
        s = voxel_list.shape

        return_volume = np.zeros((brain_mask.size,) + s[1:], dtype=voxel_list.dtype, order='C')
        return_volume[indices] = voxel_list
        vol = np.reshape(return_volume, shape3d + s[1:])

        if with_volume_dim and len(s) < 2:
            return np.expand_dims(vol, axis=3)
        return vol

    if isinstance(data, collections.Mapping):
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

    This might return points lying on the entire sphere. End-users will have to manually ensure the points to
    lie on the right hemisphere with a positive y-axis (multiply the vector by -1 if y < 0).

    .. code-block:: python

        x = sin(theta) * cos(phi)
        y = sin(theta) * sin(phi)
        z = cos(theta)

    Args:
        theta (ndarray): The matrix with the inclinations
        phi (ndarray): The matrix with the azimuths

    Returns:
        ndarray: matrix with same shape as the input (minimal two dimensions though) with on the last axis
            the [x, y, z] coordinates of each vector.
    """
    return_shape = len(theta.shape)
    if return_shape == 0:
        return_shape = 1

    def ensure_shape(coordinate):
        if len(coordinate.shape) < return_shape:
            return coordinate[..., None]
        return coordinate

    sin_theta = np.sin(theta)
    return np.stack(list(map(ensure_shape, [sin_theta * np.cos(phi),
                                            sin_theta * np.sin(phi),
                                            np.cos(theta)])), axis=return_shape)


def cartesian_to_spherical(vectors, ensure_right_hemisphere=True):
    """Create spherical coordinates (theta and phi) from the given cartesian coordinates.

    This expects a n-dimensional matrix with on the last axis a set of cartesian coordinates as (x, y, z). From that,
    this function will calculate two n-dimensional matrices for the inclinations ``theta`` and the azimuths ``phi``.

    By default the range of the output is [0, pi] for both theta and phi, meaning that the y-coordinate must
    be positive (such that all points are on the right hemisphere). For points with negative y-coordinate, this function
    will transform the coordinate to the antipodal point on the sphere and return the angles for that point. This
    behaviour can be disabled by setting ``ensure_right_hemisphere`` to false.

    Also note that this will consider the input to be unit vectors. If not, it will normalize the vectors beforehand.

    Args:
        vectors (ndarray): the n-dimensional set of cartesian coordinates (last axis should have 3 items).

    Returns:
        tuple: the matrices for theta and phi.
    """
    if ensure_right_hemisphere:
        vectors = np.copy(vectors)
        vectors[vectors[..., 1] < 0] *= -1

    vectors /= np.sqrt(np.sum(vectors**2, axis=-1))[..., None]
    vectors = np.nan_to_num(vectors)

    theta = np.arccos(vectors[..., 2])
    phi = np.arctan2(vectors[..., 1], vectors[..., 0])

    return theta, phi


def tensor_spherical_to_cartesian(theta, phi, psi):
    """Calculate the eigenvectors for a Tensor given the three angles.

    This will return the eigenvectors unsorted, since this function knows nothing about the eigenvalues. The caller
    of this function will have to sort them by eigenvalue if necessary.

    Args:
        theta (ndarray): matrix of list of theta's
        phi (ndarray): matrix of list of phi's
        psi (ndarray): matrix of list of psi's

    Returns:
        tuple: The three eigenvector for every voxel given. The return matrix for every eigenvector is of the given
        shape + [3].
    """
    v0 = spherical_to_cartesian(theta, phi)
    v1 = rotate_orthogonal_vector(v0, spherical_to_cartesian(theta + np.pi / 2.0, phi), psi)
    v2 = np.cross(v0, v1)
    return v0, v1, v2


def tensor_cartesian_to_spherical(first_eigen_vector, second_eigen_vector):
    """Compute the spherical coordinates theta, phi and psi to match the given eigen vectors.

    Only the first two eigen vectors are needed to calculate the correct angles, the last eigen vector follows
    automatically from the dot product of the first two eigen vectors.

    Since the Tensor model in MDT uses theta, phi and psi in the range [0, pi], this function can reflect the given
    eigenvalues to comply with those ranges. In particular, there are two transformations possible. The first is if
    the first eigen vector is in the left hemisphere (negative y-value), if so, it is reflected to its
    antipodal point on the right hemisphere. The second transformation is if the second eigen vector does not
    lie in the semicircle described by psi in [0, pi]. If not, the second eigen vector is reflected to
    its antipodal point within the range of psi in [0, pi].

    Args:
        first_eigen_vector (ndarray): the first eigen vectors, with on the last dimension 3 items for [x, y, z]
        second_eigen_vector (ndarray): the second eigen vectors, with on the last dimension 3 items for [x, y, z]

    Returns:
        tuple: theta, phi, psi for every voxel given.
    """
    def generate_orthogonal_vec(vec):
        """Generate the vector orthogonal to the given vector + 1/2 pi on theta.

        Instead of using spherical_to_cartesian(theta + np.pi / 2.0, phi) we can use Euclidian measures
        on the vector instead. This is faster and perhaps more precise.
        """
        denom = np.sqrt(np.sum(vec[..., (0, 1)] ** 2, axis=-1))
        result = np.stack([vec[..., 2] * vec[..., 0] / denom,
                           vec[..., 2] * vec[..., 1] / denom,
                           -np.sqrt(1 - vec[..., 2] ** 2)], axis=-1)

        # special case where x and y are both zero
        result[np.isclose(denom, 0)] = [0, 0, 1]
        return result

    def dot_product(a, b):
        return np.sum(np.multiply(a, b), axis=-1)

    right_hemisphere_e0 = np.copy(first_eigen_vector)
    right_hemisphere_e0[first_eigen_vector[..., 1] < 0] *= -1

    orthogonal_vec = generate_orthogonal_vec(right_hemisphere_e0)
    half_psi_vec = np.cross(right_hemisphere_e0, orthogonal_vec)

    correct_semicircle_e1 = np.copy(second_eigen_vector)
    correct_semicircle_e1[dot_product(half_psi_vec, correct_semicircle_e1) < 0] *= -1

    psi = np.arccos(np.clip(dot_product(orthogonal_vec, correct_semicircle_e1), -1, 1))
    theta, phi = cartesian_to_spherical(right_hemisphere_e0)

    return theta, phi, psi


def rotate_vector(basis, to_rotate, psi):
    """Uses Rodrigues' rotation formula to rotate the given vector v by psi around k.

    If a matrix is given the operation will by applied on the last dimension.

    Args:
        basis: the unit vector defining the rotation axis (k)
        to_rotate: the vector to rotate by the angle psi (v)
        psi: the rotation angle (psi)

    Returns:
        vector: the rotated vector
    """
    cross_product = np.cross(basis, to_rotate)
    dot_product = np.sum(np.multiply(basis, to_rotate), axis=-1)[..., None]
    cos_psi = np.cos(psi)[..., None]
    sin_psi = np.sin(psi)[..., None]
    return to_rotate * cos_psi + cross_product * sin_psi + basis * dot_product * (1 - cos_psi)


def rotate_orthogonal_vector(basis, to_rotate, psi):
    """Uses Rodrigues' rotation formula to rotate the given vector v by psi around k.

    If a matrix is given the operation will by applied on the last dimension.

    This function assumes that the given two vectors (or matrix of vectors) are orthogonal for every voxel.
    This assumption allows for some speedup in the rotation calculation.

    Args:
        basis: the unit vector defining the rotation axis (k)
        to_rotate: the vector to rotate by the angle psi (v)
        psi: the rotation angle (psi)

    Returns:
        vector: the rotated vector
    """
    cross_product = np.cross(basis, to_rotate)
    cos_psi = np.cos(psi)[..., None]
    sin_psi = np.sin(psi)[..., None]
    return to_rotate * cos_psi + cross_product * sin_psi


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
        def sort_versions(versions):
            versions.sort(key=lambda s: list(map(int, s.split('.'))))

        versions = os.listdir(base_path)
        sort_versions(versions)
        versions = list(reversed(versions))

        tmp_dir = tempfile.mkdtemp()

        if versions:
            previous_version = versions[0]

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
        if not os.path.exists(os.path.join(path, 'components')):
            os.makedirs(os.path.join(path, 'components'))
        cache_path = pkg_resources.resource_filename('mdt', 'data/components')

        for cache_subpath, dirs, files in os.walk(cache_path):
            subdir = cache_subpath[len(cache_path)+1:]
            config_path = os.path.join(path, 'components', subdir)

            if not os.path.exists(config_path):
                os.makedirs(config_path)

            for file in files:
                shutil.copy(os.path.join(cache_subpath, file), os.path.join(config_path, file))

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

    from mdt.lib.components import reload
    reload()

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

    if overwrite:
        # close any open files
        for handler in ModelOutputLogHandler.__instances__:
            handler.output_file = None
        if os.path.isfile(output_path):
            os.remove(output_path)

    for handler in ModelOutputLogHandler.__instances__:
        handler.output_file = output_path


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


def load_brain_mask(data_source):
    """Load a brain mask from the given data.

    Args:
        data_source (string, ndarray, tuple, nifti): Either a filename, a ndarray, a tuple as (ndarray, nifti header) or
            finally a nifti object having the method 'get_data()'.

    Returns:
        ndarray: boolean array with every voxel with a value higher than 0 set to 1 and all other values set to 0.
    """
    def _load_data():
        if isinstance(data_source, str):
            return load_nifti(data_source).get_data()
        if isinstance(data_source, np.ndarray):
            return data_source
        if isinstance(data_source, (list, tuple)):
            return np.array(data_source[0])
        if hasattr(data_source, 'get_data'):
            return data_source.get_data()
        raise ValueError('The given data source could not be recognized.')

    mask = _load_data() > 0
    if len(mask.shape) > 3:
        return mask[..., 0]
    return mask


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


def get_cl_devices(indices=None, device_type=None):
    """Get a list of all CL devices in the system.

    The indices of the devices can be used in the model fitting/sample functions for 'cl_device_ind'.

    Args:
        indices (List[int] or int): the indices of the CL devices to use. Either set this or preferred_device_type.
        device_type (str): the preferred device type, one of 'CPU', 'GPU' or 'APU'. If set, we ignore the indices
            parameter.

    Returns:
        A list of CLEnvironments, one for each device in the system.
    """
    if device_type is not None:
        return CLEnvironmentFactory.smart_device_selection(preferred_device_type=device_type)

    if indices is not None and not isinstance(indices, collections.Iterable):
        indices = [indices]

    envs = CLEnvironmentFactory.smart_device_selection()

    if indices is None:
        return envs

    return [envs[ind] for ind in indices]


def model_output_exists(model, output_folder, append_model_name_to_path=True):
    """A rudimentary check if the output for the given model exists.

    This checks if the output folder exists and contains at least the result file for each of the free parameters
    of the model.

    When using this to try to skip subjects when batch fitting it might fail if one of the models can not be calculated
    for a given subject. For example NODDI requires two shells. If that is not given we can not calculate it and
    hence no maps will be generated. When we are testing if the output exists it will therefore return False.

    Args:
        model (AbstractModel, or str): the model to check for existence.
            If a string is given the model is tried to be loaded from the components loader.
        output_folder (str): the folder where the output folder of the results should reside in
        append_model_name_to_path (boolean): by default we will append the name of the model to the output folder.
            This is to be consistent with the way the model fitting routine places the results in the
            <output folder>/<model_name> directories. Sometimes, however you might want to skip this appending.

    Returns:
        boolean: true if the output folder exists and contains files for all the parameters of the model.
    """
    if isinstance(model, str):
        model = get_model(model)()

    if append_model_name_to_path:
        output_path = os.path.join(output_folder, model.name)
    else:
        output_path = output_folder

    parameter_names = model.get_free_param_names()

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
        list of str: the path, the basename and the extension (extension includes the dot)
    """
    folder = os.path.dirname(image_path) + '/'
    basename = os.path.basename(image_path)

    for extension in ['.nii.gz', '.nii']:
        if basename[-len(extension):] == extension:
            return folder, basename[0:-len(extension)], extension
    return folder, basename, ''


def calculate_point_estimate_information_criterions(log_likelihoods, k, n):
    """Calculate various point estimate information criterions.

    These are meant to be used after maximum likelihood estimation as they assume you have a point estimate of your
    likelihood per problem.

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


def apply_mask(volumes, mask, inplace=True):
    """Apply a mask to the given input.

    Args:
        volumes (str, ndarray, list, tuple or dict): The input file path or the image itself or a list,
            tuple or dict.
        mask (str or ndarray): The filename of the mask or the mask itself
        inplace (boolean): if True we apply the mask in place on the volume image. If false we do not.

    Returns:
        Depending on the input either a single image of the same size as the input image, or a list, tuple or dict.
        This will set for all the output images the the values to zero where the mask is zero.
    """
    mask = load_brain_mask(mask)

    def apply(_volume, _mask):
        if isinstance(_volume, str):
            _volume = load_nifti(_volume).get_data()

        _mask = _mask.reshape(_mask.shape + (_volume.ndim - _mask.ndim) * (1,))

        if not inplace:
            _volume = np.copy(_volume)

        _volume[np.where(np.logical_not(_mask))] = 0
        return _volume

    if isinstance(volumes, tuple):
        return (apply(v, mask) for v in volumes)
    elif isinstance(volumes, list):
        return [apply(v, mask) for v in volumes]
    elif isinstance(volumes, dict):
        return {k: apply(v, mask) for k, v in volumes.items()}

    return apply(volumes, mask)


def apply_mask_to_file(input_fname, mask, output_fname=None):
    """Apply a mask to the given input (nifti) file.

    If no output filename is given, the input file is overwritten.

    Args:
        input_fname (str): The input file path
        mask (str or ndarray): The mask to use
        output_fname (str): The filename for the output file (the masked input file).
    """
    mask = load_brain_mask(mask)

    if output_fname is None:
        output_fname = input_fname

    write_nifti(apply_mask(input_fname, mask), output_fname, load_nifti(input_fname).header)


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


def load_sample(fname, mode='r'):
    """Load an matrix of samples from a ``.samples.npy`` file.

    This will open the samples as a numpy memory mapped array.

    Args:
        fname (str): the name of the file to load, suffix of ``.samples.npy`` is not required.
        mode (str): the mode in which to open the memory mapped sample files (see numpy mode parameter)

    Returns:
        ndarray: a memory mapped array with the results
    """
    if not os.path.isfile(fname) and not os.path.isfile(fname + '.samples.npy'):
        raise ValueError('Could not find sample results at the location "{}"'.format(fname))

    if not os.path.isfile(fname):
        fname += '.samples.npy'

    return open_memmap(fname, mode=mode)


def estimate_noise_std(input_data):
    """Estimate the noise standard deviation.

    This calculates per voxel (in the brain mas) the std over all unweighted volumes
    and takes the mean of those estimates as the standard deviation of the noise.

    The method is taken from Camino (http://camino.cs.ucl.ac.uk/index.php?n=Man.Estimatesnr).

    Args:
        input_data (mdt.lib.input_data.SimpleMRIInputData): the input data we can use to do the estimation

    Returns:
        the noise std estimated from the data. This can either be a single float, or an ndarray.

    Raises:
        :class:`~mdt.exceptions.NoiseStdEstimationNotPossible`: if the noise could not be estimated
    """
    logger = logging.getLogger(__name__)
    logger.info('Trying to estimate a noise std.')

    def all_unweighted_volumes(input_data):
        unweighted_indices = input_data.protocol.get_unweighted_indices()
        unweighted_volumes = input_data.signal4d[..., unweighted_indices]

        if len(unweighted_indices) < 2:
            raise NoiseStdEstimationNotPossible('Not enough unweighted volumes for this estimator.')

        voxel_list = create_roi(unweighted_volumes, input_data.mask)
        return np.mean(np.std(voxel_list, axis=1))

    noise_std = all_unweighted_volumes(input_data)

    if isinstance(noise_std, np.ndarray) and not is_scalar(noise_std):
        logger.info('Estimated voxel-wise noise std.')
        return noise_std

    if np.isfinite(noise_std) and noise_std > 0:
        logger.info('Estimated global noise std {}.'.format(noise_std))
        return noise_std

    raise NoiseStdEstimationNotPossible('Could not estimate a noise standard deviation from this dataset.')


class AutoDict(defaultdict):

    def __init__(self):
        """Create an auto-vivacious dictionary."""
        super().__init__(AutoDict)

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
    return mot.lib.utils.is_scalar(value)


def roi_index_to_volume_index(roi_indices, brain_mask):
    """Get the 3d index of a voxel given the linear index in a ROI created with the given brain mask.

    This is the inverse function of :func:`volume_index_to_roi_index`.

    This function is useful if you, for example, have sample results of a specific voxel
    and you want to locate that voxel in the brain maps.

    Please note that this function can be memory intensive for a large list of roi_indices

    Args:
        roi_indices (int or ndarray): the index in the ROI created by that brain mask
        brain_mask (str or 3d array): the brain mask you would like to use

    Returns:
        ndarray: the 3d voxel location(s) of the indicated voxel(s)
    """
    mask = load_brain_mask(brain_mask)
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
    if isinstance(volume_index, np.ndarray) and len(volume_index.shape) >= 2:
        return create_index_matrix(brain_mask)[volume_index[:, 0], volume_index[:, 1], volume_index[:, 2]]
    return create_index_matrix(brain_mask)[volume_index[0], volume_index[1], volume_index[2]]


def create_index_matrix(brain_mask):
    """Get a matrix with on every 3d position the linear index number of that voxel.

    This function is useful if you want to locate a voxel in the ROI given the position in the volume.

    Args:
        brain_mask (str or 3d array): the brain mask you would like to use

    Returns:
        3d ndarray: a 3d volume of the same size as the given mask and with as every non-zero element the position
            of that voxel in the linear ROI list.
    """
    mask = load_brain_mask(brain_mask)
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
    if isinstance(user_value, str):
        return user_value
    if user_value is True:
        return get_tmp_results_dir()
    return None


def create_blank_mask(volume4d_path, output_fname=None):
    """Create a blank mask for the given 4d volume.

    Sometimes you want to use all the voxels in the given dataset, without masking any voxel. Since the optimization
    routines require a mask, you have to submit one. The solution is to use a blank mask, that is, a mask that
    masks nothing.

    Args:
        volume4d_path (str): the path to the 4d volume you want to create a blank mask for
        output_fname (str): the path to the result mask. If not given, we will use the name of the input file and
            append '_mask' to it.
    """
    if not output_fname:
        input_split = split_image_path(volume4d_path)
        output_fname = input_split[0] + input_split[1] + '_mask' + input_split[2]

    volume_info = load_nifti(volume4d_path)
    mask = np.ones(volume_info.shape[:3])
    write_nifti(mask, output_fname, volume_info.header)


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
            The default is False.

    Returns:
        list of str: the list with the filenames in the order of concatenation.
    """
    images = []
    header = None

    if sort:
        volume_paths.sort(key=natural_key_sort_cb)

    for volume in volume_paths:
        nib_container = load_nifti(volume)
        header = header or nib_container.header
        image_data = nib_container.get_data()

        if len(image_data.shape) < 4:
            image_data = np.expand_dims(image_data, axis=3)

        images.append(image_data)

    combined_image = np.concatenate(images, axis=3)
    write_nifti(combined_image, output_fname, header)

    return volume_paths


def protocol_merge(protocol_paths, output_fname, sort=False):
    """Merge a list of protocols files. Writes the result as a file.

    You can enable sorting the list of protocol names based on a natural key sort. This is
    the most convenient option in the case of globbing files. By default this behaviour is disabled.

    Example usage with globbing:

    .. code-block:: python

        mdt.protocol_merge(glob.glob('*.prtcl'), 'merged.prtcl', True)

    Args:
        protocol_paths (list of str): the list with the input protocol filenames
        output_fname (str): the output filename
        sort (boolean): if true we natural sort the list of protocol files before we merge them. If false we don't.
            The default is False.

    Returns:
        list of str: the list with the filenames in the order of concatenation.
    """
    if sort:
        protocol_paths.sort(key=natural_key_sort_cb)

    protocols = list(map(load_protocol, protocol_paths))

    protocol = protocols[0]
    for i in range(1, len(protocols)):
        protocol = protocol.append_protocol(protocols[i])

    write_protocol(protocol, output_fname)
    return protocol_paths


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
    from mdt.lib.masking import create_median_otsu_brain_mask, create_write_median_otsu_brain_mask

    if output_fname:
        if not isinstance(dwi_info, (str, tuple, list)):
            raise ValueError('No header obtainable, can not write the brain mask.')
        return create_write_median_otsu_brain_mask(dwi_info, protocol, output_fname, **kwargs)
    return create_median_otsu_brain_mask(dwi_info, protocol, **kwargs)


def create_brain_mask(dwi_info, protocol, output_fname=None, **kwargs):
    """Create a brain mask.

    At the moment this uses the median-otsu algorithm, in future versions this might support better masking
    algorithms.

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
    return create_median_otsu_brain_mask(dwi_info, protocol, output_fname=output_fname, **kwargs)


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
    input_protocol = load_protocol(input_protocol)

    new_protocol = input_protocol.get_new_protocol_with_indices(volume_indices)
    write_protocol(new_protocol, output_protocol)

    input_volume = load_nifti(input_volume_fname)
    image_data = input_volume.get_data()[..., volume_indices]
    write_nifti(image_data, output_volume_fname, input_volume.header)


def natural_key_sort_cb(_str):
    """Sorting transformation to use when wanting to sorting a list using natural key sorting.

    Args:
        _str (str): the string to sort

    Returns:
        :py:func:`list`: the key to use for sorting the current element.
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', _str)]


def get_example_data(output_directory):
    """Get the MDT example data that is accompanying the installation.

    This will write the MDT example data (b1k_b2k and b6k datasets) to the indicated directory. You can use this data
    for testing MDT on your computer. These example datasets are contained within the MDT package and as such are
    available on every machine with MDT installed.

    Args:
        output_directory (str): the directory to write the files to
    """
    example_data_dir = os.path.abspath(pkg_resources.resource_filename('mdt', 'data/mdt_example_data'))
    for dataset_name in os.listdir(example_data_dir):

        dataset_output_path = os.path.join(output_directory, 'mdt_example_data', dataset_name)

        if not os.path.isdir(dataset_output_path):
            os.makedirs(dataset_output_path)

        for fname in os.listdir(os.path.join(example_data_dir, dataset_name)):
            full_fname = os.path.join(example_data_dir, dataset_name, fname)

            if os.path.isfile(full_fname):
                shutil.copy(full_fname, dataset_output_path)


def split_array_to_dict(data, param_names):
    """Create a dictionary out of an array.

    This basically splits the given nd-matrix into sub matrices based on the second dimension. The length of
    the parameter names should match the length of the second dimension. If a two dimensional matrix of shape (d, p) is
    given we return p matrices of shape (d,). If a matrix of shape (d, p, s_1, s_2, ..., s_n) is given, we return
    p matrices of shape (d, s_1, s_2, ..., s_n).

    This is basically the inverse of :func:`combine_dict_to_array`.

    Args:
        data (ndarray): a multidimensional matrix we index based on the second dimension.
        param_names (list of str): the names of the parameters, one per column

    Returns:
        dict: the results packed in a dictionary
    """
    if data.shape[1] != len(param_names):
        raise ValueError('The number of columns ({}) in the matrix does not match '
                         'the number of dictionary keys provided ({}).'.format(data.shape[1], len(param_names)))
    return {name: data[:, i, ...] for i, name in enumerate(param_names)}


def combine_dict_to_array(data, param_names):
    """Create an array out of the given data dictionary.

    The final array will consist of elements of the data dictionary, concatenated on the second dimension based
    on the order and names of the ``param_names`` list.

    This is basically the inverse of :func:`split_array_to_dict`.

    Args:
        data (dict): matrices, each of shape (n, 1) or (n,) which we will concatenate on the second dimension
        param_names (List[str]): the items we extract from the data, in that order

    Returns:
         ndarray: the dictionary elements compressed as a 2d array of size ``(n, len(param_names))``.
    """
    elements = []
    for name in param_names:
        d = data[name]
        if len(np.array(d).shape) < 2:
            d = np.transpose(np.atleast_2d([d]))
        else:
            d = np.atleast_2d(d)
        elements.append(d)
    return np.concatenate(elements, axis=1)


def covariance_to_correlation(input_maps):
    """Transform the covariance maps to correlation maps.

    This function is meant to be used on standard MDT output maps. It will look for maps named
    ``Covariance_{m0}_to_{m1}`` and ``{m[0-1]}.std`` where m0 and m1 are two map names. It will use the std. maps of m0
    and m1 to transform the covariance map into a correlation map.

    Typical use case examples (both are equal)::

        covariance_to_correlation('./BallStick_r1/')
        covariance_to_correlation(mdt.load_volume_maps('./BallStick_r1/'))

    Args:
        input_maps (dict or str): either a dictionary containing the input maps or a string with a folder name

    Returns:
        dict: the correlation maps computed from the input maps. The naming scheme is ``Correlation_{m0}_to_{m1}``.
    """
    if isinstance(input_maps, str):
        input_maps = load_volume_maps(input_maps)

    correlation_maps = {}

    for map_name in input_maps:
        match = re.match('Covariance\_(.*)\_to\_(.*)', map_name)
        if match is not None:
            m0 = match.group(1)
            m1 = match.group(2)
            if all('{}.std'.format(m) in input_maps for m in [m0, m1]):
                correlation_maps['Correlation_{}_to_{}'.format(m0, m1)] = \
                    input_maps[map_name] / (input_maps['{}.std'.format(m0)] * input_maps['{}.std'.format(m1)])
    return correlation_maps


def load_volume_maps(directory, map_names=None, deferred=True):
    """Read a number of Nifti volume maps from a directory.

    Args:
        directory (str): the directory from which we want to read a number of maps
        map_names (list or tuple): the names of the maps we want to use. If given we only use and return these maps.
        deferred (boolean): if True we return an deferred loading dictionary instead of a dictionary with the values
            loaded as arrays.

    Returns:
        dict: A dictionary with the volumes. The keys of the dictionary are the filenames (without the extension) of the
            files in the given directory.
    """
    from mdt.lib.nifti import get_all_nifti_data
    return get_all_nifti_data(directory, map_names=map_names, deferred=deferred)


def unzip_nifti(in_file, out_file=None, remove_old=False):
    """Unzip a gzipped nifti file.

    Args:
        in_file (str): the nifti file to unzip
        out_file (str): if given, the name of the output file. If not given, we will use the input filename without
            the ``.gz``.
        remove_old (boolean): if we want to remove the old (zipped) file or not
    """
    if out_file is None:
        out_file = in_file[:-3]

    with gzip.open(in_file, 'rb') as f_in:
        with open(out_file, 'wb') as f_out:
            f_out.writelines(f_in)

    if remove_old:
        os.remove(in_file)


def zip_nifti(in_file, out_file=None, remove_old=False):
    """Zip a nifti file.

    Args:
        in_file (str): the nifti file to zip
        out_file (str): if given, the name of the output file. If not given, we will use the input filename with
            ``.gz`` appended at the end.
        remove_old (boolean): if we want to remove the old (non-zipped) file or not
    """
    if out_file is None:
        out_file = in_file + '.gz'

    with open(in_file, 'rb') as f_in:
        with gzip.open(out_file, 'wb') as f_out:
            f_out.writelines(f_in)

    if remove_old:
        os.remove(in_file)


def voxelwise_vector_matrix_vector_product(a, B, c):
    """Compute the dot product of a*B*c assuming the first axii are voxel wise dimensions.

    This function can be used in error propagation where you multiply the gradient (assuming univariate function) with
    the covariance matrix with the gradient transposed.

    Args:
        a (ndarray): of size (n, m) or (x, y, z, m), vector elements per voxel
        B (ndarray): of size (n, m, m) or (x, y, z, m, m), matrix elements per voxel
        c (ndarray): of size (n, m) or (x, y, z, m), vector elements per voxel

    Returns:
        ndarray: either of size (n, 1) or of size (x, y, z, 1), the voxelwise matrix multiplication of aBc.
    """
    m = a.shape[-1]

    tmp = np.zeros_like(a)
    for ind in range(m):
        tmp[..., ind] = np.sum(a * B[..., ind, :], axis=-1)

    return np.sum(tmp * c, axis=-1)


def create_covariance_matrix(nmr_voxels, results, names, result_covars=None):
    """Create the covariance matrix for the given output maps.

    Args:
        nmr_voxels (int): the number of voxels in the output covariance matrix.
        results (dict): the results dictionary from optimization, containing the standard deviation maps
            as '<name>.std' for each of the given names. If a map is not present we will use 0 for that variance.
        names (List[str]): the names of the maps to load, the order of the names is the order of the diagonal
            elements.
        result_covars (dict): dictionary of covariance terms with the names specified as '<name>_to_<name>'.
            Since the order is undefined, this tests for <x>_to_<y> as <y>_to_<x>.

    Returns:
        ndarray: matrix of size (n, m) for n voxels and m names.
            If no covariance elements are given, we use zero for all off-diagonal terms.
    """
    m = len(names)
    covars = np.zeros((nmr_voxels, m, m)).astype(np.float64)

    for ind in range(m):
        covars[:, ind, ind] = results.get(names[ind] + '.std', 0)**2

    if result_covars:
        for x in range(m):
            for y in range(m):
                if '{}_to_{}'.format(names[x], names[y]) in result_covars:
                    covars[:, x, y] = result_covars['{}_to_{}'.format(names[x], names[y])]
                elif '{}_to_{}'.format(names[y], names[x]) in result_covars:
                    covars[:, x, y] = result_covars['{}_to_{}'.format(names[y], names[x])]
    return covars


def get_intermediate_results_path(output_dir, tmp_dir):
    """Get a temporary results path for processing.

    Args:
        output_dir (str): the output directory of the results
        tmp_dir (str): a preferred tmp dir. If not given we create a temporary directory in the output_dir.

    Returns:
        str: a path for saving intermediate computation results
    """
    if tmp_dir is None:
        return os.path.join(output_dir, DEFAULT_INTERMEDIATE_RESULTS_SUBDIR_NAME)

    if not output_dir.endswith('/'):
        output_dir += '/'

    return os.path.join(tmp_dir, hashlib.md5(output_dir.encode('utf-8')).hexdigest())


def compute_noddi_dti(model, input_data, results, noddi_d=1.7e-9):
    """Compute NODDI-like statistics from Tensor/Kurtosis parameter fits.

    Several authors noted correspondence between NODDI parameters and DTI parameters [1, 2]. This function computes
    the neurite density index (NDI) and NODDI's measure of neurite dispersion using Tensor parameters.

    Args:
        model (str or :class:`~mdt.models.base.EstimableModel`):
            The model we used to compute the results. Can be the name of a composite model or an implementation
            of a composite model.  Can be any model that uses the Tensor or KurtosisTensor compartment models.
            We need this information because we need to determine the volumes used to compute the results.
        input_data (mdt.lib.input_data.MRIInputData): the input data used for computing the model results
        results (dict): the results data, should contain at least:

            - <model>.d (ndarray): principal diffusivity
            - <model>.dperp0 (ndarray): primary perpendicular diffusion
            - <model>.dperp1 (ndarray): primary perpendicular diffusion

            And, if present, we also use these:

            - <model>.FA (ndarray): if computed already, the Fractional Anisotropy of the given diffusivities
            - <model>.MD (ndarray): if computed already, the Mean Diffusivity of the given diffusivities
            - <model>.MK (ndarray): if computing for Kurtosis, the computed Mean Kurtosis.
                If not given, we assume unity.

            Where <model> can be either 'Tensor' or 'KurtosisTensor' or the empty string in which case we take
            maps without a model name prefix.

        noddi_d (float): the intrinsic diffusivity of the intra-neurite compartment of NODDI. This is typically
            assumed to be d = 1.7x10^-9 m^2/s.

    Returns:
        dict: maps for the the NODDI-DTI, NDI and ODI measures.

    References:
        1. Edwards LJ, Pine KJ, Ellerbrock I, Weiskopf N, Mohammadi S. NODDI-DTI: Estimating neurite orientation and
            dispersion parameters from a diffusion tensor in healthy white matter.
            Front Neurosci. 2017;11(DEC):1-15. doi:10.3389/fnins.2017.00720.
        2. Lampinen B, Szczepankiewicz F, Martensson J, van Westen D, Sundgren PC, Nilsson M. Neurite density
            imaging versus imaging of microscopic anisotropy in diffusion MRI: A model comparison using spherical
            tensor encoding. Neuroimage. 2017;147(July 2016):517-531. doi:10.1016/j.neuroimage.2016.11.053.
    """
    from mdt.lib.post_processing import DTIMeasures

    def tau_to_kappa(tau):
        """Using non-linear optimization, convert the NODDI-DTI Tau variables to NODDI kappa's.

        Args:
            tau (ndarray): the list of tau's per voxel.

        Returns:
            ndarray: the list of corresponding kappa's
        """
        tau_func = SimpleCLFunction.from_string('''
            double tau(double kappa){
                if(kappa < 1e-12){
                    return 1/3.0;
                }
                return 0.5 * ( 1 / ( sqrt(kappa) * dawson(sqrt(kappa) ) ) - 1/kappa);
            }''', dependencies=[dawson()])

        objective_func = SimpleCLFunction.from_string('''
            double tau_to_kappa(local const mot_float_type* const x, void* data, local mot_float_type* objective_list){
                return pown(tau(x[0]) - ((_tau_to_kappa_data*)data)->tau, 2);
            }
        ''', dependencies=[tau_func])

        result = minimize(objective_func, np.ones_like(tau),
                          data=Struct({'tau': Array(tau, 'mot_float_type', as_scalar=True)},
                                      '_tau_to_kappa_data'),
                          use_local_reduction=False)
        kappa = result.x
        kappa[kappa > 64] = 1
        kappa[kappa < 0] = 1
        return kappa

    def get_value(name, default=None):
        data = results.get('Tensor.' + name, results.get('KurtosisTensor.' + name, results.get(name, default=default)))

        if data.ndim > 2:
            return create_roi(data, input_data.mask)
        return data

    d = get_value('d')
    dperp0 = get_value('dperp0')
    dperp1 = get_value('dperp1')

    FA = get_value('FA', default=DTIMeasures.fractional_anisotropy(d, dperp0, dperp1))
    MD = np.copy(get_value('MD', (d + dperp0 + dperp1) / 3.))

    tau = 1 / 3. * (1 + (4 / np.fabs(noddi_d - MD)) * (MD * FA / np.sqrt(3 - 2 * FA ** 2)))
    tau = np.nan_to_num(tau)

    kappa = tau_to_kappa(tau)
    odi = np.mean(np.arctan2(1.0, kappa) * 2 / np.pi, axis=1)

    if isinstance(model, str):
        model = get_model(model)()

    used_volumes = model.get_used_volumes(input_data)
    used_protocol = input_data.protocol.get_new_protocol_with_indices(used_volumes)

    shells = used_protocol.get_b_values_shells()
    if shells:
        b = shells[0]['b_value']
        if len(shells) > 1:
            b = shells[1]['b_value'] - shells[0]['b_value']

        sum = (d ** 2 + dperp0 ** 2 + dperp1 ** 2) / 5 + 2 * (d * dperp0 + d * dperp1 + dperp0 * dperp1) / 15

        MD += ((b / 6) * sum) * results.get('MK', 1)

    ndi = 1 - np.sqrt(0.5 * ((3 * MD) / noddi_d - 1))
    ndi = np.clip(np.nan_to_num(ndi), 0, 1)

    return_as_maps = results.get('Tensor.d', results.get('KurtosisTensor.d', results.get('d'))).ndim > 2

    results = {'NODDI_DTI_NDI': ndi, 'NODDI_DTI_TAU': tau, 'NODDI_DTI_KAPPA': kappa, 'NODDI_DTI_ODI': odi}
    if return_as_maps:
        return restore_volumes(results, input_data.mask)
    return results


