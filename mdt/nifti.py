import glob
import os
from contextlib import contextmanager

import nibabel as nib
import numpy as np

from mdt.deferred_mappings import DeferredActionDict

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
        :class:`nibabel.nifti1.Nifti1Image`
    """
    path = nifti_filepath_resolution(nifti_volume)
    return nib.load(path)


def load_all_niftis(directory, map_names=None):
    """Loads all niftis in the given directory as nibabel nifti files.

    This does not load the data directly, it loads the niftis in a dictionary. To get a direct handle to the image
    data use the function :func:`get_all_image_data`.

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


def get_all_image_data(directory, map_names=None, deferred=True):
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


def write_nifti(data, header, output_fname, affine=None, use_data_dtype=True, **kwargs):
    """Write data to a nifti file.

    Args:
        output_fname (str): the name of the resulting nifti file, this function will append .nii.gz if no
            suitable extension is given.
        data (ndarray): the data to write to that nifti file
        header (nibabel header): the nibabel header to use as header for the nifti file
        affine (ndarray): the affine transformation matrix
        use_data_dtype (boolean): if we want to use the dtype from the data instead of that from the header
            when saving the nifti.
        **kwargs: other arguments to Nifti1Image from NiBabel
    """
    @contextmanager
    def header_dtype():
        old_dtype = header.get_data_dtype()

        if use_data_dtype:
            dtype = data.dtype
            if data.dtype == np.bool:
                dtype = np.char

            try:
                header.set_data_dtype(dtype)
            except nib.spatialimages.HeaderDataError:
                pass

        yield header
        header.set_data_dtype(old_dtype)

    if not (output_fname.endswith('.nii.gz') or output_fname.endswith('.nii')):
        output_fname += '.nii.gz'

    with header_dtype() as header:
        nib.Nifti1Image(data, affine, header, **kwargs).to_filename(output_fname)


def write_all_as_nifti(volumes, directory, nifti_header, overwrite_volumes=True, gzip=True):
    """Write a number of volume maps to the specific directory.

    Args:
        volumes (dict): the volume maps (in 3d) with the results we want to write.
            The filenames are generated using the keys in the given volumes
        directory (str): the directory to write to
        nifti_header: the nifti header to use for each of the volumes
        overwrite_volumes (boolean): defaults to True, if we want to overwrite the volumes if they exists
        gzip (boolean): if True we write the files as .nii.gz, if False we write the files as .nii
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key, volume in volumes.items():
        extension = '.nii'
        if gzip:
            extension += '.gz'
        filename = key + extension

        full_filename = os.path.abspath(os.path.join(directory, filename))

        if os.path.exists(full_filename):
            if overwrite_volumes:
                os.remove(full_filename)
                write_nifti(volume, nifti_header, full_filename)
        else:
            write_nifti(volume, nifti_header, full_filename)


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
