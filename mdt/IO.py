import glob
import os
import numpy as np
import nibabel as nib
from mdt.deferred_mappings import DeferredActionDict

__author__ = 'Robbert Harms'
__date__ = "2014-08-28"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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


def write_nifti(data, header, output_fname, affine=None, **kwargs):
    """Write data to a nifti file.

    Args:
        output_fname (str): the name of the resulting nifti file
        data (ndarray): the data to write to that nifti file
        header (nibabel header): the nibabel header to use as header for the nifti file
        affine (ndarray): the affine transformation matrix
        **kwargs: other arguments to Nifti1Image from NiBabel

    """
    nib.Nifti1Image(data, affine, header, **kwargs).to_filename(output_fname)


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


class Nifti(object):

    @staticmethod
    def write_volume_map(name, result_volume, directory, nifti_header, overwrite_volumes=True):
        """Write a single volume as a nifti (.nii) to the given directory.

        Args:
            name (str): the name of the volume
            result_volume (ndarray): the volume we want to write out
            directory (str): the directory to write to
            nifti_header: the nifti header to use for each of the volumes
            overwrite_volumes (boolean): defaults to True, if we want to overwrite the volumes if they exists
        """
        Nifti.write_volume_maps({name: result_volume}, directory, nifti_header, overwrite_volumes)

    @staticmethod
    def write_volume_maps(result_volumes, directory, nifti_header, overwrite_volumes=True, gzip=True):
        """Write a number of maps (image result volumes) to the specific directory.

        Args:
            result_volumes (dict): the volume maps (3d) with the results we want to write out
                The naming of the file is the key of the volume with the extension appended by this function
            directory (str): the directory to write to
            nifti_header: the nifti header to use for each of the volumes
            overwrite_volumes (boolean): defaults to True, if we want to overwrite the volumes if they exists
            gzip (boolean): if True we write the files as .nii.gz, if False we write the files as .nii
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        for key, volume in result_volumes.items():
            extension = '.nii'
            if gzip:
                extension += '.gz'
            filename = key + extension

            full_filename = os.path.abspath(os.path.join(directory, filename))

            if os.path.exists(full_filename):
                if overwrite_volumes:
                    os.remove(full_filename)
                    nib.Nifti1Image(volume, None, nifti_header).to_filename(full_filename)
            else:
                nib.Nifti1Image(volume, None, nifti_header).to_filename(full_filename)

    @staticmethod
    def load_nibabel_proxies(directory, map_names=None):
        """Loads the nibabel proxies for the maps in the given directory.

        If map_names is given we will only load the given map names. Else, we load all .nii and .nii.gz files in the
        given directory.

        Args:
            directory (str): the directory from which we want to read the nibabel proxies
            map_names (list of str): the names of the maps we want to use. If given, we only use and return these maps.

        Returns:
            dict: A dictionary with the volumes. The keys of the dictionary are the filenames
                without the extension of the .nii(.gz) files in the given directory.
        """
        maps_paths = {}

        for path, map_name, extension in yield_nifti_info(directory):
            if not map_names or map_name in map_names:
                maps_paths.update({map_name: path})

        return {k: nib.load(v) for k, v in maps_paths.items()}

    @staticmethod
    def read_volume_maps(directory, map_names=None, deferred=True):
        """Read Nifti volume maps from the given directory.

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
        proxies = Nifti.load_nibabel_proxies(directory, map_names=map_names)
        if deferred:
            return DeferredActionDict(lambda _, item: item.get_data(), proxies)
        else:
            return {k: v.get_data() for k, v in proxies.items()}

    @staticmethod
    def get_image_headers(directory, map_names=None, deferred=True):
        """Read Nifti volume headers from the given directory.

        If map_names is given we will only load the given map names. Else, we load all .nii and .nii.gz files in the
        given directory.

        Args:
            directory (str): the directory from which we want to read the headers
            map_names (list of str): the names of the maps we want to use. If given, we only use and return these maps.
            deferred (boolean): if True we return an deferred loading dictionary instead of a dictionary with the values
                loaded as arrays.

        Returns:
            dict: A dictionary with the headers. The keys of the dictionary are the filenames
                without the extension of the .nii(.gz) files in the given directory.
        """
        proxies = Nifti.load_nibabel_proxies(directory, map_names=map_names)
        if deferred:
            return DeferredActionDict(lambda _, item: item.get_header(), proxies)
        else:
            return {k: v.get_header() for k, v in proxies.items()}

    @staticmethod
    def volume_names_generator(directory):
        """Get the names of the Nifti volume maps in the given directory.

        Args:
            directory (str): the directory to get the names of the available maps from

        Yields
            the volume names in the given directory
        """
        for _, map_name, _ in yield_nifti_info(directory):
            yield map_name


class TrackMark(object):
    """TrackMark is an proprietary visualization tool written by Alard Roebroeck and can be used to visualize fibre\
    directions.
    """

    @staticmethod
    def write_tvl_direction_pairs(tvl_filename, tvl_header, direction_pairs):
        """Write the given directions to TVL.

        The direction pairs should be a list with lists containing the vector and value to write. For example:
        ((vec, val), (vec1, val1), ...) up to three pairs are allowed.

        Args:
            tvl_filename (str): the filename to write to
            tvl_header (:class:`list`): the header for the TVL file. This is a list of either 4 or 10 entries.
                4 entries: [version, res, gap, offset]
                10 entries: [version, x_res, x_gap, x_offset, y_res, y_gap, y_offset, z_res, z_gap, z_offset]
            direction_pairs (list of ndarrays): The list with direction pairs, only three are used.
                This is a list with (vector, magnitude) tuples in which the vectors are 4d volumes with for
                every voxel a 3d coordinate.
        """
        direction_pairs = direction_pairs[0:3]
        dir_matrix = np.zeros(direction_pairs[0][0].shape[0:3] + (12,))
        for ind, dirs in enumerate(direction_pairs):
            dir_matrix[..., ind*3:ind*3+3] = np.ascontiguousarray(np.squeeze(dirs[0]))
            dir_matrix[..., 9 + ind] = np.ascontiguousarray(np.squeeze(dirs[1]))

        TrackMark.write_tvl_matrix(tvl_filename, tvl_header, dir_matrix)

    @staticmethod
    def write_tvl_matrix(tvl_filename, tvl_header, directions_matrix):
        """Write the given directions matrix to TVL.

        Args:
            tvl_filename: the filename to write to
            tvl_header: the header for the TVL file. This is a list of either 4 or 10 entries.
                4 entries: [version, res, gap, offset]
                10 entries: [version, x_res, x_gap, x_offset, y_res, y_gap, y_offset, z_res, z_gap, z_offset]
            directions_matrix: an 4dimensional matrix, of which the fourth dimension is of length 12.
        """
        if os.path.exists(tvl_filename):
            os.remove(tvl_filename)

        if not os.path.exists(os.path.dirname(tvl_filename)):
            os.makedirs(os.path.dirname(tvl_filename))

        open(tvl_filename, 'a').close()
        with open(tvl_filename, 'rb+') as f:
            version = np.array(tvl_header[0]).astype(np.uint16)
            version.tofile(f, '')

            if len(tvl_header) == 4:
                for i in range(3):
                    np.array(directions_matrix.shape[i]).astype(np.uint32).tofile(f, '')
                    np.array(tvl_header[1]).astype(np.float64).tofile(f, '')
                    np.array(tvl_header[2]).astype(np.float64).tofile(f, '')
                    np.array(tvl_header[3]).astype(np.float64).tofile(f, '')
            else:
                for i in range(3):
                    np.array(directions_matrix.shape[i]).astype(np.uint32).tofile(f, '')
                    np.array(tvl_header[i * 3 + 1]).astype(np.float64).tofile(f, '')
                    np.array(tvl_header[i * 3 + 2]).astype(np.float64).tofile(f, '')
                    np.array(tvl_header[i * 3 + 3]).astype(np.float64).tofile(f, '')

            directions_matrix = np.transpose(directions_matrix, (3, 2, 1, 0)).astype(np.float32).flatten('F')
            directions_matrix.tofile(f, '')

    @staticmethod
    def write_rawmaps(directory, volumes, overwrite_volumes=True):
        """Write a dictionary with volumes to the given directory.

        Args:
            directory (str): the directory to write to
            volumes (dict): an dictionary with the volume maps (3d) with the results we want to write out
                The naming of the file is the key of the volume with .rawmap appended by this function.
            overwrite_volumes (boolean): if we want to overwrite already present volumes
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)

        for key, volume in volumes.items():
            filename = key + '.rawmap'
            full_filename = os.path.abspath(os.path.join(directory, filename))

            if os.path.exists(full_filename):
                if overwrite_volumes:
                    os.remove(full_filename)
                    TrackMark.write_rawmap(full_filename, volume)
            else:
                TrackMark.write_rawmap(full_filename, volume)

    @staticmethod
    def write_rawmap(rawmap_filename, volume):
        """Write a rawmap to the given file.

        Args:
            rawmap_filename (str): The filename to write to, if not exists, it is created
                (along with extra directories). This should end on .rawmap, it not it is added.
            volume (ndarray): the volume to write. 3d or 4d. If 4d and 4th dimension is larger than 1
                additional maps are created.
        """
        if rawmap_filename[-len('.rawmap'):] != '.rawmap':
            rawmap_filename += '.rawmap'

        if os.path.exists(rawmap_filename):
            os.remove(rawmap_filename)

        if not os.path.exists(os.path.dirname(rawmap_filename)):
            os.makedirs(os.path.dirname(rawmap_filename))

        open(rawmap_filename, 'a').close()

        s = volume.shape
        if len(s) == 4:
            if s[3] == 1:
                volume = np.squeeze(volume, axis=(3,))
            else:
                subnames = rawmap_filename[0:-len('.rawmap')] + '_'
                for ind in range(volume.shape[3]):
                    TrackMark.write_rawmap(subnames + repr(ind) + '.rawmap', volume[..., ind])
                return

        with open(rawmap_filename, 'rb+') as f:
            np.array(s[0]).astype(np.uint16).tofile(f, '')
            np.array(s[1]).astype(np.uint16).tofile(f, '')
            np.array(s[2]).astype(np.uint16).tofile(f, '')

            m = np.transpose(volume, [2, 1, 0]).astype(np.float32).flatten('F')
            m.tofile(f, '')
