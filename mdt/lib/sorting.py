"""This module contains some routines for sorting volumes and lists voxel-wise.

For example, in some applications it can be desired to sort volume fractions voxel-wise over an entire volume. This
module contains functions for creating sort index matrices (determining the sort order), sorting volumes and lists
and anti-sorting volumes (reversing the sort operation based on the sort index).
"""
import collections
from copy import copy
import itertools
import numpy as np
from mdt.lib.nifti import get_all_nifti_data, load_nifti


__author__ = 'Robbert Harms'
__date__ = '2017-11-02'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def sort_orientations(data_input, weight_names, extra_sortable_maps):
    """Sort the orientations of multi-direction models voxel-wise.

    This expects as input 3d/4d volumes. Do not use this with 2d arrays.

    This can be used to sort, for example, simulations of the BallStick_r3 model (with three Sticks).
    There is no voxel-wise order over Sticks since for the model they are all equal compartments.
    However, when using optimization or ARD with sample, there is order within the compartments since the ARD is
    commonly placed on the second and third Sticks meaning these Sticks and there corresponding orientations are
    compressed to zero if they are not supported. In that case, the Stick with the primary orientation of diffusion
    has to be the first.

    This method accepts as input results from (MDT) model fitting and is able to sort all the maps belonging to
    a given set of equal compartments per voxel.

    Example::

        sort_orientations('./output/BallStick_r3',
                          ['w_stick0.w', 'w_stick1.w', 'w_stick2.w'],
                          [['Stick0.theta', 'Stick1.theta', 'Stick2.theta'],
                           ['Stick0.phi', 'Stick1.phi', 'Stick2.phi'], ...])

    Args:
        data_input (str or dict): either a directory or a dictionary containing the maps
        weight_names (iterable of str): The names of the maps we use for sorting all other maps. These will be sorted
            as well.
        extra_sortable_maps (iterable of iterable): the list of additional maps to sort. Every element in the given
            list should be another list with the names of the maps. The length of these second layer of lists should
            match the length of the ``weight_names``.

    Returns:
        dict: the sorted results in a new dictionary. This returns all input maps with some of them sorted.
    """
    if isinstance(data_input, str):
        input_maps = get_all_nifti_data(data_input)
        result_maps = input_maps
    else:
        input_maps = data_input
        result_maps = copy(input_maps)

    weight_names = list(weight_names)
    sortable_maps = copy(extra_sortable_maps)
    sortable_maps.append(weight_names)

    sort_index_matrix = create_sort_matrix([input_maps[k] for k in weight_names], reversed_sort=True)

    for sortable_map_names in sortable_maps:
        sorted = dict(zip(sortable_map_names, sort_volumes_per_voxel([input_maps[k] for k in sortable_map_names],
                                                                     sort_index_matrix)))
        result_maps.update(sorted)

    return result_maps


def create_sort_matrix(input_volumes, reversed_sort=False):
    """Create an index matrix that sorts the given input on the 4th dimension from small to large values (per element).

    Args:
        input_volumes (ndarray or list): either a list with 3d volumes (or 4d with a singleton on the fourth dimension),
            or a 4d volume to use directly.
        reversed_sort (boolean): if True we reverse the sort and we sort from large to small.

    Returns:
        ndarray: a 4d matrix with on the 4th dimension the indices of the elements in sorted order.
    """
    def load_maps(map_list):
        tmp = []
        for data in map_list:
            if isinstance(data, str):
                data = load_nifti(data).get_data()

            if len(data.shape) < 4:
                data = data[..., None]

            if data.shape[3] > 1:
                raise ValueError('Can not sort input volumes where one has more than one items on the 4th dimension.')

            tmp.append(data)
        return tmp

    if isinstance(input_volumes, collections.Sequence):
        maps_to_sort_on = load_maps(input_volumes)
        input_4d_vol = np.concatenate([m for m in maps_to_sort_on], axis=3)
    else:
        input_4d_vol = input_volumes

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
        input_volumes (:class:`list`): list of 4d ndarray
        sort_matrix (ndarray): 4d ndarray with for every voxel the sort index

    Returns:
        :class:`list`: the same input volumes but then with every voxel sorted according to the given sort index.
    """
    def load_maps(map_list):
        tmp = []
        for data in map_list:
            if isinstance(data, str):
                data = load_nifti(data).get_data()

            if len(data.shape) < 4:
                data = data[..., None]

            tmp.append(data)
        return tmp

    input_volumes = load_maps(input_volumes)

    if input_volumes[0].shape[3] > 1:
        volume = np.concatenate([np.reshape(m, m.shape[0:3] + (1,) + (m.shape[3],)) for m in input_volumes], axis=3)
        grid = np.ogrid[[slice(x) for x in volume.shape]]
        sorted_volume = volume[list(grid[:-2]) + [np.reshape(sort_matrix, sort_matrix.shape + (1,))] + list(grid[-1])]
        return [sorted_volume[..., ind, :] for ind in range(len(input_volumes))]
    else:
        volume = np.concatenate([m for m in input_volumes], axis=3)
        sorted_volume = volume[list(np.ogrid[[slice(x) for x in volume.shape]][:-1])+[sort_matrix]]
        return [np.reshape(sorted_volume[..., ind], sorted_volume.shape[0:3] + (1,))
                for ind in range(len(input_volumes))]


def undo_sort_volumes_per_voxel(input_volumes, sort_matrix):
    """Undo the voxel-wise sorting of volumes based on the original sort matrix.

    This uses the original sort matrix to place the elements back into the original order. For example, suppose we had
    data [a, b, c] with sort matrix [1, 2, 0] then the new results are [b, c, a]. This function will, given the
    sort matrix [1, 2, 0] and results [b, c, a] return the original matrix [a, b, c].

    Args:
        input_volumes (:class:`list`): list of 4d ndarray
        sort_matrix (ndarray): 4d ndarray with for every voxel the sort index

    Returns:
        :class:`list`: the same input volumes but then with every voxel anti-sorted according to the given sort index.
    """
    results = [np.zeros_like(vol) for vol in input_volumes]

    shape = input_volumes[0].shape

    for x, y, z in itertools.product(range(shape[0]), range(shape[1]), range(shape[2])):
        for ind, sort_ind in enumerate(list(sort_matrix[x, y, z])):
            results[sort_ind][x, y, z] = input_volumes[ind][x, y, z]

    return results

