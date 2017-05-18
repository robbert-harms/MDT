import os

import numpy as np

from mdt import load_nifti
from mdt.nifti import get_all_image_data

__author__ = 'Robbert Harms'
__date__ = "2017-02-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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
            tvl_header (:class:`list` or tuple): the header for the TVL file. This is a list of either 4 or 10 entries.
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
    def auto_convert(input_folder, model_name=None, output_folder=None, conversion_profile=None):
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
            conversion_profile (TrackMarkConversionProfile): the conversion profile to use for the conversion.
                By default it will be autodetected based on the directory name.
        """
        output_folder = output_folder or os.path.join(input_folder, 'trackmark')
        model_name = model_name or os.path.basename(os.path.normpath(input_folder))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        conversion_profile = conversion_profile or get_trackmark_conversion_profile(model_name)
        direction_pairs, volumes = conversion_profile.get_info(input_folder)
        tvl_header = conversion_profile.get_tvl_header(input_folder)

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
            tuple: the (direction_pairs, volumes) tuple. The first should contain the direction pairs
                needed for the TVL output, the second is the list of volumes to write as rawmaps.
        """
        raise NotImplementedError()

    def get_tvl_header(self, input_folder):
        """Get the TVL header we use for the TVL output.

        Args:
            input_folder (str): the folder containing the niftis to convert

        Returns:
            tuple: the tvl header data
        """
        raise NotImplementedError()


class _TMCP_BallStick_r1(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        maps_to_convert = ['FS', 'w_ball.w', 'w_stick.w']

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)

        vector_directions = [load_nifti(os.path.join(input_folder, 'Stick.vec0')).get_data()]
        vector_magnitudes = [load_nifti(os.path.join(input_folder, 'w_stick.w')).get_data() * 1e-3]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes

    def get_tvl_header(self, input_folder):
        return 1, load_nifti(os.path.join(input_folder, 'Stick.vec0')).get_header().get_zooms()[0], 0, 0


class _TMCP_BallStick_r2(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        import mdt

        maps_to_convert = ['FS', 'w_ball.w']

        sort_index_matrix = mdt.create_sort_matrix([os.path.join(input_folder, 'w_stick{}.w.nii.gz'.format(i))
                                                 for i in range(2)], reversed_sort=True)
        sorted_weights = mdt.sort_maps([os.path.join(input_folder, 'w_stick{}.w.nii.gz'.format(i)) for i in range(2)],
                                       sort_index_matrix=sort_index_matrix)
        sorted_vecs = mdt.sort_maps([os.path.join(input_folder, 'Stick{}.vec0.nii.gz'.format(i)) for i in range(2)],
                                    sort_index_matrix=sort_index_matrix)

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)
        volumes.update(dict(zip(['w_stick{}.w'.format(i) for i in range(2)], sorted_weights)))

        vector_directions = sorted_vecs
        vector_magnitudes = [v * 1e-3 for v in sorted_weights]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes

    def get_tvl_header(self, input_folder):
        return 1, load_nifti(os.path.join(input_folder, 'Stick0.vec0')).get_header().get_zooms()[0], 0, 0


class _TMCP_BallStick_r3(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        import mdt

        maps_to_convert = ['FS', 'w_ball.w']

        sort_index_matrix = mdt.create_sort_matrix([os.path.join(input_folder, 'w_stick{}.w.nii.gz'.format(i))
                                                    for i in range(3)], reversed_sort=True)
        sorted_weights = mdt.sort_maps([os.path.join(input_folder, 'w_stick{}.w.nii.gz'.format(i)) for i in range(3)],
                                       sort_index_matrix=sort_index_matrix)
        sorted_vecs = mdt.sort_maps([os.path.join(input_folder, 'Stick{}.vec0.nii.gz'.format(i)) for i in range(3)],
                                    sort_index_matrix=sort_index_matrix)

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)
        volumes.update(dict(zip(['w_stick{}.w'.format(i) for i in range(3)], sorted_weights)))

        vector_directions = sorted_vecs
        vector_magnitudes = [v * 1e-3 for v in sorted_weights]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes

    def get_tvl_header(self, input_folder):
        return 1, load_nifti(os.path.join(input_folder, 'Stick0.vec0')).get_header().get_zooms()[0], 0, 0


class _TMCP_CHARMED_r1(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        maps_to_convert = ['FR', 'Tensor.FA', 'w_hin0.w']

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)

        vector_directions = [load_nifti(os.path.join(input_folder, 'CHARMEDRestricted0.vec0')).get_data()]
        vector_magnitudes = [load_nifti(os.path.join(input_folder, 'w_res0.w')).get_data() * 1e-3]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes

    def get_tvl_header(self, input_folder):
        return 1, load_nifti(os.path.join(input_folder, 'CHARMEDRestricted0.vec0')).get_header().get_zooms()[0], 0, 0


class _TMCP_CHARMED_r2(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        import mdt

        maps_to_convert = ['FR', 'Tensor.FA', 'w_hin0.w']

        sort_index_matrix = mdt.create_sort_matrix([os.path.join(input_folder, 'w_res{}.w.nii.gz'.format(i))
                                                    for i in range(2)], reversed_sort=True)
        sorted_weights = mdt.sort_maps([os.path.join(input_folder, 'w_res{}.w.nii.gz'.format(i)) for i in range(2)],
                                       sort_index_matrix=sort_index_matrix)
        sorted_vecs = mdt.sort_maps([os.path.join(input_folder, 'CHARMEDRestricted{}.vec0.nii.gz'.format(i))
                                     for i in range(2)], sort_index_matrix=sort_index_matrix)

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)
        volumes.update(dict(zip(['CHARMEDRestricted{}.w'.format(i) for i in range(2)], sorted_weights)))

        vector_directions = sorted_vecs
        vector_magnitudes = [v * 1e-3 for v in sorted_weights]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes

    def get_tvl_header(self, input_folder):
        return 1, load_nifti(os.path.join(input_folder, 'CHARMEDRestricted0.vec0')).get_header().get_zooms()[0], 0, 0


class _TMCP_CHARMED_r3(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        import mdt

        maps_to_convert = ['FR', 'Tensor.FA', 'w_hin0.w']

        sort_index_matrix = mdt.create_sort_matrix([os.path.join(input_folder, 'w_res{}.w.nii.gz'.format(i))
                                                    for i in range(3)], reversed_sort=True)
        sorted_weights = mdt.sort_maps([os.path.join(input_folder, 'w_res{}.w.nii.gz'.format(i)) for i in range(3)],
                                       sort_index_matrix=sort_index_matrix)
        sorted_vecs = mdt.sort_maps([os.path.join(input_folder, 'CHARMEDRestricted{}.vec0.nii.gz'.format(i))
                                     for i in range(3)], sort_index_matrix=sort_index_matrix)

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)
        volumes.update(dict(zip(['CHARMEDRestricted{}.w'.format(i) for i in range(3)], sorted_weights)))

        vector_directions = sorted_vecs
        vector_magnitudes = [v * 1e-3 for v in sorted_weights]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes

    def get_tvl_header(self, input_folder):
        return 1, load_nifti(os.path.join(input_folder, 'CHARMEDRestricted0.vec0')).get_header().get_zooms()[0], 0, 0


class _TMCP_NODDI(TrackMarkConversionProfile):

    def get_info(self, input_folder):
        maps_to_convert = ['NDI', 'NODDI_EC.kappa', 'ODI', 'w_csf.w', 'w_ec.w', 'w_ic.w']

        volumes = get_all_image_data(input_folder, map_names=maps_to_convert, deferred=True)

        vector_directions = [load_nifti(os.path.join(input_folder, 'NODDI_IC.vec0')).get_data()]
        vector_magnitudes = [load_nifti(os.path.join(input_folder, 'w_ic.w')).get_data() * 1e-3]

        direction_pairs = list(zip(vector_directions, vector_magnitudes))[:3]

        return direction_pairs, volumes

    def get_tvl_header(self, input_folder):
        return 1, load_nifti(os.path.join(input_folder, 'NODDI_IC.vec0')).get_header().get_zooms()[0], 0, 0


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

    def get_tvl_header(self, input_folder):
        return 1, load_nifti(os.path.join(input_folder, 'Tensor.sorted_vec0')).get_header().get_zooms()[0], 0, 0
