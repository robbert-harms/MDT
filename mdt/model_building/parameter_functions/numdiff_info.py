__author__ = 'Robbert Harms'
__date__ = '2017-10-23'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class NumDiffInfo:
    """Encapsulates information necessary for numerical differentiation of a parameter."""

    @property
    def max_step(self):
        """Get the maximum numerical differentiation step size for this parameter.

        Returns:
            float: the maximum step size for the parameter
        """
        raise NotImplementedError()

    @property
    def scaling_factor(self):
        """A scaling factor for the parameter to ensure that it's magnitude is close to unitary.

        This should return a value such that when a parameter is scaled with this value, the value is close to
        unitary.

        Returns:
            float: a scaling factor to use for the parameter
        """
        raise NotImplementedError()

    @property
    def use_bounds(self):
        """If we should use the boundary conditions for this parameter when calculating a derivative.

        This is typically set to True unless we are dealing with circular parameters.

        Returns:
            boolean: if we need to use the lower and upper bounds (defined in the parameter) when calculating
                the derivatives for this model.
        """
        raise NotImplementedError()


class SimpleNumDiffInfo(NumDiffInfo):

    def __init__(self, max_step=0.1, scale_factor=1, use_bounds=True, use_lower_bound=True, use_upper_bound=True):
        """A basic implementation of the numerical differentiation info for a parameter.

        Args:
            max_step (float): the numerical differentiation step size
            scale_factor (float): a scaling factor to rescale the parameter a unitary range
            use_bounds (boolean): if we need to use the boundary condition for this parameter
            use_lower_bound (boolean): if we are using bounds, if we are using the lower bound
            use_upper_bound (boolean): if we are using bounds, if we are using the upper bound
        """
        self._numdiff_step = max_step
        self._scale_factor = scale_factor
        self._use_bounds = use_bounds
        self._use_lower_bound = use_lower_bound
        self._use_upper_bound = use_upper_bound

    @property
    def max_step(self):
        return self._numdiff_step

    @property
    def scaling_factor(self):
        return self._scale_factor

    @property
    def use_bounds(self):
        return self._use_bounds

    @property
    def use_lower_bound(self):
        return self._use_lower_bound

    @property
    def use_upper_bound(self):
        return self._use_upper_bound
