import glob
import os
import numpy as np
from mdt.utils import restore_volumes
from mdt.lib.nifti import write_nifti, load_nifti

__author__ = 'Robbert Harms'
__date__ = "2017-02-28"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def volume_map_npy_to_nifti(npy_fname, nifti_header, nifti_fname=None):
    """Convert a volume-map npy file to a nifti file.

    Args:
        npy_fname (str): the filename of the npy file to load
        nifti_header (nibabel header): the header file for the nifti
        nifti_fname (str): the filename of the nifti file. If not given it defaults to the same directory as the
            npy file.
    """
    data = np.load(npy_fname, mmap_mode='r')

    if nifti_fname is None:
        nifti_fname = os.path.join(os.path.dirname(npy_fname),
                                   os.path.splitext(os.path.basename(npy_fname))[0] + '.nii.gz')
    write_nifti(data, nifti_fname, nifti_header)


def load_all_npy_files(directory):
    """Load all the npy files in the given directory.

    Args:
        directory (str): the directory to load the npy files in

    Returns:
        dict: the loaded npy files with as keys the filename (without the .npy extension) and as value the
            memory mapped array
    """
    file_names = list(map(lambda p: os.path.splitext(os.path.basename(p))[0],
                          glob.glob(os.path.join(directory, '*.npy'))))

    results_dict = {}
    for file_name in file_names:
        results_dict[file_name] = np.load(os.path.join(directory, file_name + '.npy'), mmap_mode='r')
    return results_dict


def samples_npy_to_nifti(samples_npy_fname, used_mask, nifti_header, nifti_fname=None):
    """Convert a npy file containing sample results to a nifti file.

    Since the sample npy files are stored as a two dimensional matrix (with on the first axis the ROI index number
    and on the second the samples), we need to have the lookup table for the spatial information about the samples.

    Args:
        samples_npy_fname (str): the filename of the samples file to convert
        used_mask (ndarray or str): either an three dimensional matrix with the mask or a path to a nifti file.
        nifti_header (nibabel header): the header to use for writing the nifti file
        nifti_fname (str): the filename of the nifti file. If not given it defaults to the same directory as the
            samples file.
    """
    samples = np.load(samples_npy_fname, mmap_mode='r')

    if isinstance(used_mask, str):
        used_mask = load_nifti(used_mask).get_data()

    if np.count_nonzero(used_mask) != samples.shape[0]:
        raise ValueError(
            'The number of voxels in the mask ({}) does not correspond '
            'with the number of voxels in the samples file ({})'.format(np.count_nonzero(used_mask), samples.shape[0]))

    if nifti_fname is None:
        nifti_fname = os.path.join(os.path.dirname(samples_npy_fname),
                                   os.path.splitext(os.path.basename(samples_npy_fname))[0] + '.nii.gz')

    volume = restore_volumes(samples, used_mask)
    write_nifti(volume, nifti_fname, nifti_header)
