import glob
import os

import itertools
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
        """Write a single volume to the given directory.

        Args:
            name: the name of the volume
            result_volume: the volume we want to write out
            directory: the directory to write to
            nifti_header: the nifti header to use for each of the volumes
            overwrite_volumes: defaults to True, if we want to overwrite the volumes if they exists
        """
        Nifti.write_volume_maps({name: result_volume}, directory, nifti_header, overwrite_volumes)

    @staticmethod
    def write_volume_maps(result_volumes, directory, nifti_header, overwrite_volumes=True):
        """Write a number of maps (image result volumes) to the specific directory.

        Args:
            result_volumes: an dictionary with the volume maps (3d) with the results we want to write out
                The naming of the file is the key of the volume with .nii.gz appended by this function
            directory: the directory to write to
            nifti_header: the nifti header to use for each of the volumes
            overwrite_volumes: defaults to True, if we want to overwrite the volumes if they exists
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        for key, volume in result_volumes.items():
            filename = key + '.nii.gz'
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
            directory: the directory from which we want to read a number of maps
            map_names: the names of the maps we want to load. If given we only load and return these maps.

        Returns:
            A dictionary with the volumes. The keys of the dictionary are the filenames
            (without the extension) of the files in the given directory.
        """
        maps = {}
        for extension in ('.nii', '.nii.gz'):
            for f in glob.glob(os.path.join(directory, '*' + extension)):
                map_name = os.path.basename(f)[0:-len(extension)]

                if map_names is None or map_name in map_names:
                    d = nib.load(f).get_data()
                    s = d.shape
                    if len(s) > 3 and s[3] == 1:
                        d = np.squeeze(d, axis=(3,))

                    maps.update({map_name: d})
        return maps

    @staticmethod
    def volume_names_generator(directory):
        """Get the names of the Nifti volume maps in the given directory.

        Args:
            directory: the directory to get the names of the available maps from

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
    def write_tvl_direction_pairs(tvl_filename, tvl_header, direction_pairs, vector_ranking=None,
                                  direction_scalar=None):
        """Write the given directions to TVL.

        The direction pairs should be a list with lists containing the vector and value to write. For example:
            ((vec, val), (vec1, val1), ...) up to three pairs are allowed.

        Args:
            tvl_filename (str): the filename to write to
            tvl_header (list): the header for the TVL file. This is a list of either 4 or 10 entries.
                4 entries: [version, res, gap, offset]
                10 entries: [version, x_res, x_gap, x_offset, y_res, y_gap, y_offset, z_res, z_gap, z_offset]
            direction_pairs (list of ndarrays): The list with direction pairs, only three are used.
            vector_ranking (list): the list of map names in the same order as the eigen values/vectors that determine
                per voxel the ranking of the vectors/values.
            direction_scalar (float): the amount by which we scale the direction length before write out
        """
        if vector_ranking is not None:
            if len(vector_ranking) < len(direction_pairs):
                raise ValueError('Not enough vector rankings provided. We have {0} eigen '
                                 'pairs and only {1} ranking maps.'.format(len(direction_pairs), len(vector_ranking)))
            dir_matrix = TrackMark.generate_dir_matrix_ordered(direction_pairs, vector_ranking)
        else:
            dir_matrix = TrackMark.generate_dir_matrix_unordered(direction_pairs)

        direction_scalar = direction_scalar or 1
        dir_matrix[..., 9:] *= direction_scalar

        TrackMark.write_tvl_matrix(tvl_filename, tvl_header, dir_matrix)

    @staticmethod
    def generate_dir_matrix_unordered(direction_pairs):
        direction_pairs = direction_pairs[0:3]
        dir_matrix = np.zeros(direction_pairs[0][0].shape[0:3] + (12,))
        for ind, dirs in enumerate(direction_pairs):
            dir_matrix[..., ind*3:ind*3+3] = np.ascontiguousarray(np.squeeze(dirs[0]))
            dir_matrix[..., 9 + ind] = np.ascontiguousarray(np.squeeze(dirs[1]))
        return dir_matrix

    @staticmethod
    def generate_dir_matrix_ordered(direction_pairs, vector_ranking):
        ranking = np.argsort(np.concatenate([vr[..., None] for vr in vector_ranking], axis=3), axis=3)

        direction_pairs = direction_pairs[0:3]
        dir_matrix = np.zeros(direction_pairs[0][0].shape[0:3] + (12,))

        shape3d = direction_pairs[0][0].shape
        for l_x, l_y, l_z in itertools.product(range(shape3d[0]), range(shape3d[1]), range(shape3d[2])):

            pair_ranking = ranking[l_x, l_y, l_z]
            for linear_ind, ranking_ind in enumerate(pair_ranking):
                chosen_vector = direction_pairs[ranking_ind][0]
                chosen_val = direction_pairs[ranking_ind][1]

                dir_matrix[l_x, l_y, l_z, linear_ind*3:linear_ind*3+3] = chosen_vector[l_x, l_y, l_z]
                dir_matrix[l_x, l_y, l_z, linear_ind + 9] = chosen_val[l_x, l_y, l_z]

        return dir_matrix

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
