import six
import numpy as np
import numbers
from mdt.nifti import load_nifti

__author__ = 'Robbert Harms'
__date__ = "2015-08-25"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def autodetect_noise_std_loader(data_source):
    """A function to get a noise std using the given data source.

    This tries to do auto detecting for the following data sources:

        - None: return 1
        - double: uses the given single value for all voxels
        - ndarray: use a value per voxel (this should not be a roi list, it should be an actual volume
            of the same size as the dataset)
        - string (and not 'auto'): a filename we will try to parse as a noise std
        - the string 'auto': try to estimate the noise std

    Args:
        data_source: the data source from which to get a noise std

    Returns:
        NoiseStdLoader: a noise std loader instance.
    """
    if isinstance(data_source, NoiseStdLoader):
        return data_source

    elif data_source is None:
        return NoiseEstimationLoader()

    elif isinstance(data_source, numbers.Number):
        return SingleValueNoiseStd(data_source)

    elif isinstance(data_source, np.ndarray):
        return VoxelWiseNoiseStd(data_source)

    elif isinstance(data_source, six.string_types):
        return LoadNoiseFromFile(data_source)

    raise ValueError('The given data source could not be recognized.')


class NoiseStdLoader(object):
    """Interface for loading a noise std from different sources."""

    def get_noise_std(self, problem_data):
        """The public method for getting the noise std from this loader.

        Args:
            problem_data (:class:`~mdt.utils.DMRIProblemData`): the dmri problem data in use.
                Some loaders might need this for loading the noise std.

        Returns:
            noise std, either a single value or an ndarray with a value per voxel
        """


class NoiseEstimationLoader(NoiseStdLoader):

    def __init__(self):
        """A loader that estimates the noise std from the problem data"""

    def get_noise_std(self, problem_data):
        from mdt.utils import estimate_noise_std
        return estimate_noise_std(problem_data)


class SingleValueNoiseStd(NoiseStdLoader):

    def __init__(self, noise_std):
        """Returns the given noise std"""
        self._noise_std = noise_std

    def get_noise_std(self, problem_data):
        return self._noise_std


class VoxelWiseNoiseStd(NoiseStdLoader):

    def __init__(self, noise_std_map):
        """Returns a noise std map with one value per voxel."""
        self._noise_std_map = noise_std_map

    def get_noise_std(self, problem_data):
        return self._noise_std_map


class LoadNoiseFromFile(NoiseStdLoader):

    def __init__(self, file_name):
        """Load a noise std from a file.

        This will try to detect if the given file is a text file with a single noise std, or if it is a nifti / map file
        with a voxel wise noise std.
        """
        self._file_name = file_name

    def get_noise_std(self, problem_data):
        if self._file_name[-4:] == '.txt':
            with open(self._file_name, 'r') as f:
                return float(f.read())

        return load_nifti(self._file_name).get_data()
