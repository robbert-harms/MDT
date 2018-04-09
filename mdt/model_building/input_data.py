__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class InputData(object):
    """A simple container for the input data for optimization/sampling models."""

    def get_input_data(self, parameter_name):
        """Get the input data for the given parameter.

        Args:
             parameter_name (str): the name of the parameter for which we want to get input data

        Returns:
            float, ndarray or None: either a scalar, a vector or a matrix with values for the given parameter.
                None should be returned if no suitable value can be found.
        """
        raise NotImplementedError()

    @property
    def nmr_problems(self):
        """Get the number of problems present in this input data.

        Returns:
            int: the number of problem instances
        """
        raise NotImplementedError()

    @property
    def nmr_observations(self):
        """Get the number of observations/data points per problem.

        The minimum is one observation sper problem.

        Returns:
            int: the number of instances per problem (aka data points)
        """
        raise NotImplementedError()

    @property
    def observations(self):
        """Return the observations stored in this input data container.

        Returns:
            ndarray: The list of observed instances per problem. Should be a 2d matrix of type float with as
                columns the observations and as rows the problems.
        """
        raise NotImplementedError()

    @property
    def noise_std(self):
        """The noise standard deviation we will use during model evaluation.

        During optimization or sampling the model will be evaluated against the observations using a
        likelihood function. Most of these likelihood functions need a standard deviation representing the noise
        in the data.

        Returns:
            number of ndarray: either a scalar or a 2d matrix with one value per problem instance.
        """
        raise NotImplementedError()
