import glob
import gzip
import os
import copy
import nibabel as nib
import numpy as np
import shutil

from mdt.lib.deferred_mappings import DeferredActionDict

__author__ = 'Robbert Harms'
__date__ = "2014-08-28"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def load_nifti(nifti_volume):
    """Load and return a nifti file.

    This will apply path resolution if a filename without extension is given. See the function
    :func:`nifti_filepath_resolution` for details.

    Args:
        nifti_volume (string): The filename of the volume to use.

    Returns:
        :class:`nibabel.nifti2.Nifti2Image`
    """
    path = nifti_filepath_resolution(nifti_volume)
    return nifti_info_decorate_nibabel_image(nib.load(path))


def load_all_niftis(directory, map_names=None):
    """Loads all niftis in the given directory as nibabel nifti files.

    This does not load the data directly, it loads the niftis in a dictionary. To get a direct handle to the image
    data use the function :func:`get_all_nifti_data`.

    If ``map_names`` is given we will only load the given maps. Else, we will load all .nii and .nii.gz files.
    The map name is the filename of a nifti without the extension.

    In the case both an .nii and a .nii.gz with the same name exists we will load the .nii as the main map
    and the .nii.gz with its extension.

    Args:
        directory (str): the directory from which we want to load the niftis
        map_names (list of str): the names of the maps we want to use. If given, we only use and return these maps.

    Returns:
        dict: A dictionary with the loaded nibabel proxies (see :func:`load_nifti`).
            The keys of the dictionary are the filenames without the extension of the .nii(.gz) files
            in the given directory.
    """
    maps_paths = {}

    for path, map_name, extension in yield_nifti_info(directory):
        if not map_names or map_name in map_names:
            if map_name in maps_paths:
                map_name += extension
            maps_paths.update({map_name: path})

    return {k: load_nifti(v) for k, v in maps_paths.items()}


def get_all_nifti_data(directory, map_names=None, deferred=True):
    """Get the data of all the nifti volumes in the given directory.

    If map_names is given we will only load the given map names. Else, we load all .nii and .nii.gz files in the
    given directory.

    Args:
        directory (str): the directory from which we want to read a number of maps
        map_names (list of str): the names of the maps we want to use. If given, we only use and return these maps.
        deferred (boolean): if True we return an deferred loading dictionary instead of a dictionary with the values
            loaded as arrays.

    Returns:
        dict: A dictionary with the volumes. The keys of the dictionary are the filenames
            without the extension of the .nii(.gz) files in the given directory.
    """
    proxies = load_all_niftis(directory, map_names=map_names)
    if deferred:
        return DeferredActionDict(lambda _, item: item.get_data(), proxies)
    else:
        return {k: v.get_data() for k, v in proxies.items()}


def write_nifti(data, output_fname, header=None, affine=None, use_data_dtype=True, **kwargs):
    """Write data to a nifti file.

    This will write the output directory if it does not exist yet.

    Args:
        data (ndarray): the data to write to that nifti file
        output_fname (str): the name of the resulting nifti file, this function will append .nii.gz if no
            suitable extension is given.
        header (nibabel header): the nibabel header to use as header for the nifti file. If None we will use
            a default header.
        affine (ndarray): the affine transformation matrix
        use_data_dtype (boolean): if we want to use the dtype from the data instead of that from the header
            when saving the nifti.
        **kwargs: other arguments to Nifti2Image from NiBabel
    """
    if header is None:
        header = nib.nifti2.Nifti2Header()

    if use_data_dtype:
        header = copy.deepcopy(header)
        dtype = data.dtype
        if data.dtype == np.bool:
            dtype = np.char
        try:
            header.set_data_dtype(dtype)
        except nib.spatialimages.HeaderDataError:
            pass

    if not (output_fname.endswith('.nii.gz') or output_fname.endswith('.nii')):
        output_fname += '.nii.gz'

    if not os.path.exists(os.path.dirname(output_fname)):
        os.makedirs(os.path.dirname(output_fname))

    if isinstance(header, nib.nifti2.Nifti2Header):
        format = nib.Nifti2Image
    else:
        format = nib.Nifti1Image

    format(data, affine, header=header, **kwargs).to_filename(output_fname)


def write_all_as_nifti(volumes, directory, nifti_header=None, overwrite_volumes=True, gzip=True):
    """Write a number of volume maps to the specific directory.

    Args:
        volumes (dict): the volume maps (in 3d) with the results we want to write.
            The filenames are generated using the keys in the given volumes
        directory (str): the directory to write to
        nifti_header: the nifti header to use for each of the volumes.
        overwrite_volumes (boolean): defaults to True, if we want to overwrite the volumes if they exists
        gzip (boolean): if True we write the files as .nii.gz, if False we write the files as .nii
    """
    for key, volume in volumes.items():
        extension = '.nii'
        if gzip:
            extension += '.gz'
        filename = key + extension

        full_filename = os.path.abspath(os.path.join(directory, filename))

        if os.path.exists(full_filename):
            if overwrite_volumes:
                os.remove(full_filename)
                write_nifti(volume, full_filename, header=nifti_header)
        else:
            write_nifti(volume, full_filename, header=nifti_header)


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
    raise ValueError('No nifti file could be found using the path {}'.format(file_path))


def yield_nifti_info(directory):
    """Get information about the nifti volumes in the given directory.

    Args:
        directory (str): the directory to get the names of the available maps from

    Yields:
        tuple: (path, map_name, extension) for every map found
    """
    for extension in ('.nii', '.nii.gz'):
        for f in glob.glob(os.path.join(directory, '*' + extension)):
            yield f, os.path.basename(f)[0:-len(extension)], extension


def is_nifti_file(file_name):
    """Check if the given file is a nifti file.

    This only checks if the extension of the given file ends with .nii or with .nii.gz

    Args:
        file_name (str): the name of the file

    Returns:
        boolean: true if the file looks like a nifti file, false otherwise
    """
    return file_name.endswith('.nii') or file_name.endswith('.nii.gz')


def unzip_nifti(input_filename, output_filename):
    """Unzips the given nifti file.

    This will create the output directories if they do not yet exist.

    Args:
        input_filename (str): the nifti file we would like to unzip. Should have the extension ``.gz``.
        output_filename (str): the location for the output file. Should have the extension ``.nii``.

    Raises:
        ValueError: if the extensions of either the input or output filename are not correct.
    """
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))

    if not input_filename.rstrip().endswith('.gz') or not output_filename.rstrip().endswith('.nii'):
        raise ValueError('The input filename should have extension ".gz" and the '
                         'output filename should have extension ".nii".')

    with gzip.open(input_filename, 'rb') as f_in, open(output_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


class NiftiInfo:

    def __init__(self, header=None, filepath=None):
        """A nifti information object to store meta data alongside an array.

        Args:
            header: the nibabel nifti header
            filepath (str): the on-disk filepath of the corresponding nifti file
        """
        self.header = header
        self.filepath = filepath


class NiftiInfoDecorated:
    """The additional type of an array after it has been subclassed by :func:`nifti_info_decorate_array`.

    This subclass can be used to check if an array has nifti info attached to it.
    """

    @property
    def nifti_info(self):
        """Get the nifti information attached to the subclass.

        Returns:
            NiftiInfo: the nifti information object
        """
        raise NotImplementedError()


def nifti_info_decorate_array(array, nifti_info=None):
    """Decorate the provided numpy array with nifti information.

    This can be used to ensure that an array is of additional subclass :class:`NiftiInfoDecorated`.

    Args:
        array (ndarray): the numpy array to decoreate
        nifti_info (NiftiInfo): the nifti info to attach to the array
    """
    class NiftiInfoDecoratedArray(type(array), NiftiInfoDecorated):
        def __new__(cls, input_array, nifti_info=None):
            """Decorate an existing input array with some additional about a nifti file.

            This is typically used to store the original nifti header and the filepath as an special attribute of
            the array. Please note that this metadata does not survive subarrays and views of this array.

            Args:
                input_array (ndarray): the array we decorate
                nifti_info (NiftiInfo): a nifti information object

            Returns:
                ndarray: the decorated array
            """
            obj = input_array.view(cls)
            obj._nifti_info = nifti_info or NiftiInfo()
            return obj

        @property
        def nifti_info(self):
            return self._nifti_info

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self._nifti_info = getattr(obj, '_nifti_info', None)

    return NiftiInfoDecoratedArray(array, nifti_info)


def nifti_info_decorate_nibabel_image(nifti_obj):
    """Decorate the nibabel image container such that the ``get_data`` method returns a NiftiInfoDecorated ndarray.

    Args:
        nifti_obj: a nibabel nifti object
    """
    original_function = nifti_obj.get_data

    def get_data(self, *args, **kwargs):
        data = original_function(*args, **kwargs)
        return nifti_info_decorate_array(data, NiftiInfo(header=self.header, filepath=self.get_filename()))

    nifti_obj.get_data = get_data.__get__(nifti_obj, type(nifti_obj))
    return nifti_obj
