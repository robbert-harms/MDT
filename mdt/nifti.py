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
    :func:`nifti_filepath_resolution` for details.

    Args:
        nifti_volume (string): The filename of the volume to use.

    Returns:
        :class:`nibabel.nifti1.Nifti1Image`
    """
    path = nifti_filepath_resolution(nifti_volume)
    return nib.load(path)


def load_all_niftis(directory, map_names=None):
    """Loads all niftis in the given directory.

    If map_names is given we will only load the given maps. Else, we load all .nii and .nii.gz files in the
    given directory. The map name is the filename of a nifti without the extension.

    Args:
        directory (str): the directory from which we want to load the niftis
        map_names (list of str): the names of the maps we want to use. If given, we only use and return these maps.

    Returns:
        dict: A dictionary with the loaded nibabel proxies (see :func:`load_nifti`).
            The keys of the dictionary are the filenames without the extension of the .nii(.gz) files
            in the given directory.
    """
    maps_paths = {}

    for path, map_name, _ in yield_nifti_info(directory):
        if not map_names or map_name in map_names:
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



class TrackMark(object):
    """TrackMark is an proprietary visualization tool written by Alard Roebroeck and can be used to visualize fibre\
    directions. This class is meant to convert nifti files to TrackMark specific files.
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
            dir_matrix[..., ind*3:ind*3+3] = np.ascontiguousarray(TrackMark._ensure_3d(np.squeeze(dirs[0])))
            dir_matrix[..., 9 + ind] = np.ascontiguousarray(TrackMark._ensure_3d(np.squeeze(dirs[1])))

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

    @staticmethod
    def auto_convert(input_folder, model_name=None, output_folder=None):
        """Convert the nifti files in the given folder to Trackmark.

        This automatically loads the correct files based on the model name. This is normally the dirname of the given
        path. If that is not the case you can give the model name explicitly.

        By default it outputs the results to a folder named "trackmark" in the given input folder. This can of course
        be overridden using the output_folder parameter.

        Args:
            input_folder (str): the name of the input folder
            model_name (str): the name of the model, if not given we use the last dirname of the given path
            output_folder (str): the output folder, if not given we will output to a subfolder "trackmark" in the
                given directory.
        """
        output_folder = output_folder or os.path.join(input_folder, 'trackmark')
        model_name = model_name or os.path.basename(os.path.normpath(input_folder))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        conversion_profile = get_trackmark_conversion_profile(model_name)
        direction_pairs, volumes = conversion_profile.get_info(input_folder)

        tvl_header = (1, 1.8, 0, 0)

        TrackMark.write_rawmaps(output_folder, volumes)
        TrackMark.write_tvl_direction_pairs(output_folder + '/master.tvl', tvl_header, direction_pairs)

    @staticmethod
    def _ensure_3d(array):
        if len(array.shape) < 3:
            return array[..., None]
        return array


def get_trackmark_conversion_profile(model_name):
    """Get the TrackMark conversion profile for the given model name.

    Args:
        model_name (str): the name of the model

    Returns:
        TrackMarkConversionProfile: the trackmark conversion profile for the given model name

    Raises:
        ValueError: if no conversion profile for the given model name could be found.
    """
    if '_TMCP_' + model_name.replace('-ExVivo', '') in globals():
        return globals()['_TMCP_' + model_name.replace('-ExVivo', '')]()

    raise ValueError('No trackmark conversion profile could be found for the model named {}.'.format(model_name))


class TrackMarkConversionProfile(object):

    def get_info(self, input_folder):
        """Create the trackmark files (TVL and rawmap) using this profile for the data in the given input folder

        Args:
            input_folder (str): the folder containing the niftis to convert

        Returns:
            tuple: the (direction_paris, volumes) tuple. The first should contain the direction pairs
                needed for the TVL output, the second is the list of volumes to write as rawmaps.
        """


class _TMCP_BallStick_r1(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        maps_to_convert = ['FS', 'w_ball.w', 'w_stick.w']

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)

        vector_directions = [load_nifti(os.path.join(input_folder, 'Stick.vec0')).get_data()]
        vector_magnitudes = [load_nifti(os.path.join(input_folder, 'w_stick.w')).get_data() * 1e-2]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes


class _TMCP_BallStick_r2(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        import mdt

        maps_to_convert = ['FS', 'w_ball.w']

        sort_output = mdt.sort_maps(
            [os.path.join(input_folder, 'w_stick{}.w.nii.gz'.format(i)) for i in range(2)],
            extra_maps_to_sort=[os.path.join(input_folder, 'Stick{}.vec0.nii.gz'.format(i)) for i in range(2)],
            reversed_sort=True)

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)
        volumes.update(dict(zip(['w_stick{}.w'.format(i) for i in range(2)], sort_output[0])))

        vector_directions = sort_output[1]
        vector_magnitudes = [v * 1e-2 for v in sort_output[0]]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes


class _TMCP_BallStick_r3(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        import mdt

        maps_to_convert = ['FS', 'w_ball.w']

        sort_output = mdt.sort_maps(
            [os.path.join(input_folder, 'w_stick{}.w.nii.gz'.format(i)) for i in range(3)],
            extra_maps_to_sort=[os.path.join(input_folder, 'Stick{}.vec0.nii.gz'.format(i)) for i in range(3)],
            reversed_sort=True)

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)
        volumes.update(dict(zip(['w_stick{}.w'.format(i) for i in range(3)], sort_output[0])))

        vector_directions = sort_output[1]
        vector_magnitudes = [v * 1e-2 for v in sort_output[0]]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes


class _TMCP_CHARMED_r1(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        maps_to_convert = ['FR', 'Tensor.FA', 'w_hin0.w']

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)

        vector_directions = [load_nifti(os.path.join(input_folder, 'CHARMEDRestricted0.vec0')).get_data()]
        vector_magnitudes = [load_nifti(os.path.join(input_folder, 'w_res0.w')).get_data() * 1e-2]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes


class _TMCP_CHARMED_r2(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        import mdt

        maps_to_convert = ['FR', 'Tensor.FA', 'w_hin0.w']

        sort_output = mdt.sort_maps(
            [os.path.join(input_folder, 'w_res{}.w.nii.gz'.format(i)) for i in range(2)],
            extra_maps_to_sort=[os.path.join(input_folder, 'CHARMEDRestricted{}.vec0.nii.gz'.format(i))
                                for i in range(2)],
            reversed_sort=True)

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)
        volumes.update(dict(zip(['CHARMEDRestricted{}.w'.format(i) for i in range(2)], sort_output[0])))

        vector_directions = sort_output[1]
        vector_magnitudes = [v * 1e-2 for v in sort_output[0]]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes


class _TMCP_CHARMED_r3(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        import mdt

        maps_to_convert = ['FR', 'Tensor.FA', 'w_hin0.w']

        sort_output = mdt.sort_maps(
            [os.path.join(input_folder, 'w_res{}.w.nii.gz'.format(i)) for i in range(3)],
            extra_maps_to_sort=[os.path.join(input_folder, 'CHARMEDRestricted{}.vec0.nii.gz'.format(i))
                                for i in range(3)],
            reversed_sort=True)

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)
        volumes.update(dict(zip(['CHARMEDRestricted{}.w'.format(i) for i in range(3)], sort_output[0])))

        vector_directions = sort_output[1]
        vector_magnitudes = [v * 1e-2 for v in sort_output[0]]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes


class _TMCP_NODDI(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        maps_to_convert = ['NDI', 'NODDI_EC.kappa', 'ODI', 'w_csf.w', 'w_ec.w', 'w_ic.w']

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)

        vector_directions = [load_nifti(os.path.join(input_folder, 'NODDI_IC.vec0')).get_data()]
        vector_magnitudes = [load_nifti(os.path.join(input_folder, 'w_ic.w')).get_data() * 1e-2]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes


class _TMCP_Tensor(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        maps_to_convert = ['Tensor.FA', 'Tensor.AD', 'Tensor.MD', 'Tensor.RD',
                           'Tensor.sorted_eigval0', 'Tensor.sorted_eigval1', 'Tensor.sorted_eigval2']

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)

        vector_directions = [load_nifti(os.path.join(input_folder, 'Tensor.sorted_vec{}'.format(i))).get_data()
                             for i in range(3)]
        vector_magnitudes = [load_nifti(os.path.join(input_folder, 'Tensor.sorted_eigval{}'.format(i))).get_data() * 1e6
                             for i in range(3)]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes
