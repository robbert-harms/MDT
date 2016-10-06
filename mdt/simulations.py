"""This module contains some functions that allow for generating simulated data.

The simulated data is on the level of diffusion MRI models, not on the level of simulated physical molecule interaction
as found in for example Camino.
"""

import numbers
import os
import nibabel as nib
import numpy as np
import mdt
from mdt.components_loader import NoiseSTDCalculatorsLoader
from mdt.utils import MockDMRIProblemData
from mot.cl_routines.mapping.calculate_model_estimates import CalculateModelEstimates

__author__ = 'Robbert Harms'
__date__ = "2016-03-17"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def create_parameters_cube(primary_parameter_index, randomize_parameter_indices, grid_size,
                           default_values, lower_bounds, upper_bounds, dtype=np.float32, seed=None):
    """Create a simple 3d parameters cube.

    On the first dimension we put a linearly spaced primary parameter and on the second dimension we randomly change
    the other indicated parameters. The 3d dimension holds the parameter realizations for the other dimensions.

    Args:
        primary_parameter_index (int): the index of the primary parameter for the values on the first axis
        randomize_parameter_indices (list of int): the indices of the parameter we sample randomly using a uniform
            distribution on the half open interval between the [lower, upper) bounds. See np.random.uniform.
        grid_size (tuple of int): the size of the generated grid, the first value refers to the first dimension, the
            second to the second dimension.
        default_values (list of float): the default values for each of the parameters in the model
        lower_bounds (list of float): the lower bounds used for the generation of the grid
        upper_bounds (list of float): the upper bounds used for the generation of the grid
        dtype (dtype): the numpy data type for this grid
        seed (int): if given the seed for the random number generator, this makes the random parameters predictable.

    Returns:
        ndarray: a three dimensional cube for the parameters
    """
    grid = permutate_parameters([primary_parameter_index], default_values, lower_bounds, upper_bounds, grid_size[0],
                                dtype=dtype)
    grid = np.reshape(grid, (grid.shape[0], 1, grid.shape[1]))
    grid = np.repeat(grid, grid_size[1], axis=1)

    random_state = np.random.RandomState(seed)

    for param_ind in randomize_parameter_indices:
        grid[:, :, param_ind] = random_state.uniform(lower_bounds[param_ind], upper_bounds[param_ind], size=grid_size)

    return grid


def add_noise_realizations(signal_cube, nmr_noise_realizations, noise_sigma, seed=None):
    """Add noise realizations to a signal cube.

    The given signal cube should be a 3d matrix that contains on the last axis the signals per protocol line. All
    the other axis normally represent the variations of parameter values that generated the last dimension.

    This function inserts a new 3th dimension to that matrix, one to contain the variation over noise realisations. It
    will then make the signals rician distributed per index on that new 3th dimension.

    Args:
        signal_cube (ndarray): the 3d matrix with the signals
        nmr_noise_realizations (int): the number of noise realizations to use on the new third axis
        noise_sigma (float): the noise level, given by:
            noise_level = unweighted_signal_height / SNR
        seed (int): if given, the seed for the random number generation

    Returns:
        ndarray: a 4d cube with on the newly added third dimension the variating noise realizations
    """
    signals = np.reshape(signal_cube, signal_cube.shape[0: 2] + (1, signal_cube.shape[2]))
    signals = np.repeat(signals, nmr_noise_realizations, axis=2)
    return make_rician_distributed(signals, noise_sigma, seed=seed)


def simulate_signals_param_cube(model_name, protocol, parameters_cube):
    """Generate the signal for the given model for a generated parameters cube.

    Args:
        model_name (str): the name of the model we want to generate the values for
        protocol (Protocol): the protocol object we use for generating the signals
        parameters_cube (ndarray): the 3d matrix with the parameters for every problem instance

    Returns:
        signal estimates as a cube
    """
    parameters = np.reshape(parameters_cube, (-1, parameters_cube.shape[-1]))
    simulated_signals = simulate_signals(model_name, protocol, parameters)
    return np.reshape(simulated_signals, parameters_cube.shape[0:2] + (simulated_signals.shape[-1], ))


def permutate_parameters(var_params_ind, default_values, lower_bounds, upper_bounds, grid_size,
                         dtype=np.float32):
    """Generate the combination of parameters for a simulation.

    This is useful if you want to generate a list of different parameter combinations. You do not need to use
    this if you want to simulate only one parameter.

    This generates for each of the parameters of interest a linearly indexed range of parameter values
    starting with the lower bounds and ending at the upper bound (both inclusive). The length of the list is determined
    by the grid size per parameter. Next we create a matrix with the cartesian product of each of these parameters
    of interest and with all the other parameters set to their default value.

    Args:
        var_params_ind (list of int): the list of indices into the parameters. This indices the parameters
            we want to vary.
        default_values (list of float): the default values for each of the parameters in the model
        lower_bounds (list of float): the lower bounds used for the generation of the grid
        upper_bounds (list of float): the upper bounds used for the generation of the grid
        grid_size (int or list of int): the size of the grid. If a single int is given we assume a grid
            equal in all dimensions. If a list is given it should match the number of variable parameter indices
            and should contain a grid size for each parameter.
        dtype (dtype): the data type of the result matrix

    Returns:
        ndarray: the matrix with all combinations of the parameters of interest and with all other parameters set to
            the given default value.
    """
    if isinstance(grid_size, numbers.Number):
        grid_size = [int(grid_size)] * len(var_params_ind)

    result = np.reshape(default_values, [len(lower_bounds), 1]).astype(dtype)

    repeat_mult = 1
    for linear_ind, params_ind in enumerate(var_params_ind):
        result = np.tile(result, grid_size[linear_ind])
        result[params_ind] = np.repeat(np.linspace(lower_bounds[params_ind],
                                                   upper_bounds[params_ind],
                                                   grid_size[linear_ind]), repeat_mult)
        repeat_mult *= grid_size[linear_ind]

    return np.transpose(result)


def get_permuted_indices(nmr_var_params, grid_size):
    """Get for every parameter of interest the locations per parameter value.

    This is useful if you want to generate a list of different parameter combinations. You do not need to use
    this if you want to simulate only one parameter.

    Suppose you have three variable parameters and you generate all permutations using permutate_parameters(), then you
    might want to know for any given parameter and for any value of that parameter at which indices that parameter
    occurs. This function tells you where.

    Note, we could have taken the nmr_var_params from the grid size, but the grid size can be a single scalar for all
    params.

    Args:
        nmr_var_params (int): the number of variable parameters
        grid_size (int or list of int): the grid size for all or per parameter

    Returns:
        ndarray: per permutation the value index indexing the parameter value
    """
    indices = np.zeros((nmr_var_params, 1), dtype=np.int64)

    repeat_mult = 1
    for ind in range(nmr_var_params):
        indices = np.tile(indices, grid_size[ind])
        indices[ind, :] = np.repeat(np.arange(0, grid_size[ind]), repeat_mult)
        repeat_mult *= grid_size[ind]

    return np.transpose(indices)


def simulate_signals(model_name, protocol, parameters):
    """Generate the signal for the given model for each of the parameters.

    This function only accepts a 2d list of parameters. For a generated parameters cube use function
    simulate_signals_param_cube.

    Args:
        model_name (str): the name of the model we want to generate the values for
        protocol (Protocol): the protocol object we use for generating the signals
        parameters (ndarray): the 2d matrix with the parameters for every problem instance

    Returns:
        signal estimates
    """
    problem_data = MockDMRIProblemData(protocol, None, None, None)

    model = mdt.get_model(model_name)
    model.set_problem_data(problem_data)

    signal_evaluate = CalculateModelEstimates()
    return signal_evaluate.calculate(model, parameters)


def make_rician_distributed(signals, noise_level, seed=None):
    """Make the given signal Rician distributed.

    To calculate the noise level divide the signal of the unweighted volumes by the SNR you want. For example,
    for a unweighted signal b0=1e4 and a desired SNR of 20, you need an noise level of 1e4/20 = 500.

    Args:
        signals: the signals to make Rician distributed
        noise_level: the level of noise to add. The actual Rician stdev depends on the signal. See ricestat in
            the mathworks library. The noise level can be calculated using b0/SNR.
        seed (int): if given, the seed for the random number generation

    Returns:
        ndarray: Rician distributed signals.
    """
    random_state = np.random.RandomState(seed)
    x = noise_level * random_state.normal(size=signals.shape) + signals
    y = noise_level * random_state.normal(size=signals.shape)
    return np.sqrt(np.power(x, 2), np.power(y, 2)).astype(signals.dtype)


def list_2d_to_4d(item_list):
    """Convert a 2d signal/parameter list to a 4d volume.

    This appends two singleton volumes to the signal list to make it 4d.

    Args:
         item_list (2d ndarray): the list with on the first dimension every problem and on the second
            the signals per protocol line.

    Returns:
        ndarray: 4d ndarray of size (1, 1, n, p) where n is the number of problems and p the length of the protocol.
    """
    return np.reshape(item_list, (1, 1) + item_list.shape)


def save_data_volume(file_name, data):
    """Save the 3d/4d volume to the given file.

    Args:
        file_name (str): the output file name. If the directory does not exist we create one.
        data (ndarray): the 4d array to save.
    """
    if not os.path.isdir(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    img = nib.Nifti1Image(data, np.eye(4))
    img.to_filename(file_name)


def save_2d_list_as_4d_volume(file_name, data):
    """Save the given 2d list with values as a 4d volume.

    This is a convenience function that calls list_2d_to_4d and volume4d_to_file after each other.

    Args:
        file_name (str): the output file name. If the directory does not exist we create one.
        data (ndarray): the 2d array to save
    """
    save_data_volume(file_name, list_2d_to_4d(data))


def get_unweighted_volumes(signals, protocol):
    """Get the signals and protocol for only the unweighted signals.

    Args:
        signals (ndarray): the matrix with for every problem (first dimension) the volumes (second dimension)
        protocol (Protocol): the protocol object

    Returns:
        tuple: unweighted signals and the protocol for only the unweighted indices.
    """
    unweighted_indices = protocol.get_unweighted_indices()

    unweighted_signals = signals[:, unweighted_indices]
    unweighted_protocol = protocol.get_new_protocol_with_indices(unweighted_indices)

    return unweighted_signals, unweighted_protocol


def estimate_noise_std(simulated_noisy_signals, protocol, noise_estimator_name='AllUnweightedVolumes'):
    """Estimate the noise on the noisy simulated dataset.

    This routine tries to estimate the noise level of the added noise. It first fits an S0 model to the data with
    a noise std of 1. It then removes this estimated S0 from the given signal and tries to estimate the noise std
    on the result.

    Args:
        simulated_noisy_signals (ndarray): the list with per problem the noisy simulated signal
        protocol (Protocol): the protocol object
        noise_estimator_name (str): the name of the noise estimator to use

    Returns:
        float: the noise standard deviation
    """
    mask = np.ones(simulated_noisy_signals.shape[0:3])

    loader = NoiseSTDCalculatorsLoader()
    cls = loader.get_class(noise_estimator_name)
    calculator = cls(mdt.load_problem_data(simulated_noisy_signals, protocol, mask))

    return calculator.estimate()
