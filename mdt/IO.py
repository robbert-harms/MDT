import glob
import os
import numpy as np
import nibabel as nib
import scipy.io


__author__ = 'Robbert Harms'
__date__ = "2014-08-28"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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
    def read_volume_maps(directory, map_names=None):
        """Read a number of Nifti volume maps that were written using write_volume_maps.

        Args:
            directory (str): the directory from which we want to read a number of maps
            map_names (list of str): the names of the maps we want to use. If given we only use and return these maps.

        Returns:
            dict: A dictionary with the volumes. The keys of the dictionary are the filenames
                without the extension of the .nii(.gz) files in the given directory.
        """
        maps = {}
        for extension in ('.nii', '.nii.gz'):
            for f in glob.glob(os.path.join(directory, '*' + extension)):
                map_name = os.path.basename(f)[0:-len(extension)]

                if map_names is None or map_name in map_names:
                    maps.update({map_name: nib.load(f).get_data()})
        return maps

    @staticmethod
    def volume_names_generator(directory):
        """Get the names of the Nifti volume maps in the given directory.

        Args:
            directory (str): the directory to get the names of the available maps from

        Returns:
            generator: this yields the volume names in the given directory
        """
        for extension in ('.nii', '.nii.gz'):
            for f in glob.glob(os.path.join(directory, '*' + extension)):
                yield os.path.basename(f)[0:-len(extension)]


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


class Matlab(object):

    @staticmethod
    def load_maps_from_mat(mat_file):
        """Load a result dictionary from the given matlab mat file

        Args:
            mat_file (str): The location of the .mat file.

        Returns:
            dict: All the items in the mat file by their name.
        """
        mat_data = scipy.io.loadmat(mat_file)
        fields = (e[0] for e in scipy.io.whosmat(mat_file))
        results = {}
        for field in fields:
            results.update({field: mat_data[field]})
        return results
