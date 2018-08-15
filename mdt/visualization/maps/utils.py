import glob
import os

from mdt.lib.nifti import nifti_filepath_resolution
from mdt.utils import split_image_path


def get_shortest_unique_names(paths):
    """Get the shortest unique map name between two or more nifti file paths.

    This function is useful when loading multiple maps like for example ``./foo.nii`` and ``./foo.nii.gz`` or
    ``../directory0/foo.nii`` and ``../directory1/foo.nii``. In all cases the map name (foo) is similar,
    but we want to be able to show them distinctly. This function tries to find the shortest unique map names
    for each of the maps.

    Example output:
    * [``./foo.nii``] -> ``foo``
    * [``./foo.nii``, ``./foo.nii.gz``] -> [``foo.nii``, ``foo.nii.gz``]
    * [``../directory0/foo.nii``, ``../directory1/foo.nii``] -> [``directory0/foo``, ``directory1/foo``]

    Args:
        paths (list of str): the paths to the different nifti files

    Returns:
        tuple: the map names for the given set of paths (in the same order)
    """
    if not len(paths):
        return []

    real_paths = []
    for path in paths:
        if os.path.exists(path):
            real_paths.append(path)
        else:
            real_paths.append('/' + path)

    paths = real_paths

    dirs, names, exts = zip(*map(split_image_path, paths))

    if len(set(paths)) == 1:
        return names

    new_names = []

    multiple_dirs = len(set(dirs)) > 1
    common_prefix = os.path.commonpath(dirs)

    def multiple_maps_in_same_dir(current_ind, current_directory, current_name):
        for ind, (directory, name, ext, full_path) in enumerate(zip(dirs, names, exts, paths)):
            if ind != current_ind and directory == current_directory and name == current_name:
                return True
        return False

    for ind, (directory, name, ext, full_path) in enumerate(zip(dirs, names, exts, paths)):
        if multiple_dirs:
            new_name = os.path.relpath(directory, common_prefix) + '/' + name
            if new_name.startswith('../') or new_name.startswith('..\\'):
                new_name = new_name[3:]
            if common_prefix == '/' or common_prefix == '\\':
                new_name = '/' + new_name
        else:
            new_name = name

        if multiple_maps_in_same_dir(ind, directory, name):
            new_name += ext

        new_names.append(new_name)

    return new_names


def find_all_nifti_files(paths):
    """Find the paths to the nifti files in the given paths.

    For directories we add all the nifti files from that directory, for file we try to resolve them to nifti files.

    Args:
        paths (list of str): the list of file paths

    Returns:
        list of str: the list of all nifti files (no more directories)
    """
    def find_niftis(paths, recurse=True):
        niftis = []
        for path in paths:
            if os.path.isdir(path):
                if recurse:
                    niftis.extend(find_niftis(glob.glob(path + '/*.nii*'), recurse=False))
            else:
                try:
                    niftis.append(nifti_filepath_resolution(path))
                except ValueError:
                    pass
        return niftis
    return find_niftis(paths)
