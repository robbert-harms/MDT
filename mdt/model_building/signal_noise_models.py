from mot.lib.cl_function import SimpleCLFunctionParameter
from .parameters import FreeParameter
from .model_functions import SimpleModelCLFunction
from .parameter_functions.transformations import CosSqrClampTransform

__author__ = 'Robbert Harms'
__date__ = "2014-08-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SignalNoiseModel(SimpleModelCLFunction):
    """Signal noise models can add noise to the signal resulting from the model.

    They require the signal resulting from the model and zero or more parameters and they return a new signal
    with noise added. This should have a model signature like:

    .. code-block:: c

        double fname(double signal, <noise model parameters ...>);

    For example, if the noise model has only one parameter 'sigma' the function should look like:

    .. code-block:: c

        double fname(double signal, double sigma);

    The CL function should return a single double that represents the signal with the signal noise
        added to it.
    """


class JohnsonNoise(SignalNoiseModel):

    def __init__(self):
        """Johnson noise adds noise to the signal using the formula:

        .. code-block:: c

            sqrt(signal^2 + eta^2)

        """
        super().__init__(
            'double', 'JohnsonNoise', [
                SimpleCLFunctionParameter('double', 'signal'),
                FreeParameter('mot_float_type', 'eta', False, 0.1, 0, 100, parameter_transform=CosSqrClampTransform())],
            'return hypot(signal, (double)eta);')
