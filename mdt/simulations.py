import numpy as np
import collections
from mdt.lib.components import get_model
from mdt.lib.nifti import get_all_nifti_data
from mdt.utils import create_roi, restore_volumes, MockMRIInputData
from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Array, Zeros

__author__ = 'Robbert Harms'
__date__ = '2017-05-29'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def create_signal_estimates(model, input_data, parameters):
    """Create the signals estimates for your estimated model parameters.

    This function is typically used to obtain signal estimates from optimization results.

    This function evaluates the model as it is in the model fitting and sample. That is, this method includes
    the gradient deviations (if set in the input data) and loads all static and fixed parameters maps.

    Args:
        model (str or model): the model or the name of the model to use for estimating the signals
        input_data (mdt.utils.MRIInputData): the input data object, we will set this to the model
        parameters (str or dict): either a directory file name or a dictionary containing optimization results
            Each element is assumed to be a 4d volume with the voxels we are using for the simulations.

    Returns:
        ndarray: the 4d array with the signal estimates per voxel
    """
    if isinstance(model, str):
        model = get_model(model)()

    model.set_input_data(input_data)
    build_model = model.build()

    if isinstance(parameters, str):
        parameters = get_all_nifti_data(parameters)

    parameters = create_roi(parameters, input_data.mask)
    parameters = model.param_dict_to_array(parameters)

    kernel_data = {'data': build_model.get_kernel_data(),
                   'parameters': Array(parameters, ctype='mot_float_type'),
                   'estimates': Zeros((parameters.shape[0], build_model.get_nmr_observations()), 'mot_float_type')}

    _get_simulate_function(build_model).evaluate(kernel_data, parameters.shape[0])
    results = kernel_data['estimates'].get_data()

    return restore_volumes(results, input_data.mask)


def simulate_signals(model, protocol, parameters):
    """Estimate the signals of a given model for the given combination of protocol and parameters.

    In contrast to the function :func:`create_signal_estimates`, this function does not incorporate the gradient
    deviations. Furthermore, this function expects a two dimensional list of parameters and this function will
    simply evaluate the model for each set of parameters.

    Args:
        model (str or model): the model or the name of the model to use for estimating the signals
        protocol (mdt.protocols.Protocol): the protocol we will use for the signal simulation
        parameters (dict or ndarray): the parameters for which to simulate the signal. It can either be a matrix with
            for every row every model parameter, or a dictionary with for every parameter a 1d array.

    Returns:
        ndarray: a 2d array with for every parameter combination the simulated model signal
    """
    if isinstance(model, str):
        model = get_model(model)()

    model.set_input_data(MockMRIInputData(protocol=protocol))
    build_model = model.build()

    if isinstance(parameters, collections.Mapping):
        parameters = model.param_dict_to_array(parameters)

    nmr_problems = parameters.shape[0]

    kernel_data = {'data': build_model.get_kernel_data(),
                   'parameters': Array(parameters, ctype='mot_float_type'),
                   'estimates': Zeros((nmr_problems, build_model.get_nmr_observations()), 'mot_float_type')
                   }

    _get_simulate_function(build_model).evaluate(kernel_data, nmr_problems)
    return kernel_data['estimates'].get_data()


def add_rician_noise(signals, noise_level, seed=None):
    """Make the given signal Rician distributed.

    To calculate the noise level divide the signal of the unweighted volumes by the SNR you want. For example,
    for a unweighted signal b0=1e4 and a desired SNR of 20, you need an noise level of 1e4/20 = 500.

    Args:
        signals: the signals to make Rician distributed
        noise_level: the level of noise to add. The actual Rician stdev depends on the signal. See ricestat in
            the mathworks library. The noise level can be calculated using b0/SNR.
        seed (int): if given, the seed for the random number generation

    Returns:
        ndarray: make every element of the input signals contain Rician distributed noise.
    """
    random_state = np.random.RandomState(seed)
    x = noise_level * random_state.normal(size=signals.shape) + signals
    y = noise_level * random_state.normal(size=signals.shape)
    return np.sqrt(np.power(x, 2), np.power(y, 2)).astype(signals.dtype)


def _get_simulate_function(model):
    """Get the simulation function.

    This is a wrapper around the model evaluation function. As such, it includes the protocol adaptation callbacks.
    It creates the model signal just as used by the log-likelihood function during model optimization.
    """
    eval_function_info = model.get_model_eval_function()
    return SimpleCLFunction.from_string('''
        void simulate(void* data, local mot_float_type* parameters, global mot_float_type* estimates){
            for(uint i = 0; i < ''' + str(model.get_nmr_observations()) + '''; i++){
                estimates[i] = ''' + eval_function_info.get_cl_function_name() + '''(data, parameters, i);
            }
        }
    ''', dependencies=[eval_function_info])

