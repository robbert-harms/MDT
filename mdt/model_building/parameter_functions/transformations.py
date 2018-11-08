import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2014-06-20"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractTransformation:
    """The transformations define the encode and decode operations needed to build a codec.

    These objects define the transformation to and from model and optimization space.
    """

    def get_cl_encode(self):
        """Get the CL encode assignment constructor

        Returns
            AssignmentConstructor: The cl code assignment constructor for encoding the parameter.
        """
        raise NotImplementedError()

    def get_cl_decode(self):
        """Get the CL decode assignment constructor

        Returns:
            AssignmentConstructor: The cl code assignment constructor for decoding the parameter.
        """
        raise NotImplementedError()

    def encode_bounds(self, lower_bounds, upper_bounds):
        """Encode the given bounds into the encoded parameter space.

        Args:
            lower_bounds (floar or ndarray): either a single bound, or per voxel a bound
            upper_bounds (floar or ndarray): either a single bound, or per voxel a bound

        Returns:
            tuple: the transformed lower and upper bounds, same shape as inputs
        """
        raise NotImplementedError()


class AssignmentConstructor:

    def create_assignment(self, parameter_variable, lower_bound, upper_bound):
        """Create the assignment string.

        Args:
            parameter_variable (str): the name of the parameter variable holding the current value in the kernel
            lower_bound (str): the value or the name of the variable holding the value for the lower bound
            upper_bound (str): the value or the name of the variable holding the value for the upper bound

        Returns:
            str: the transformation assignment
        """
        raise NotImplementedError()


class FormatAssignmentConstructor(AssignmentConstructor):

    def __init__(self, assignment):
        """Assignment constructor that formats the given assignment template.

        This expects that the assignment string has elements like:

        * ``{parameter_variable}``: for the parameter variable
        * ``{lower_bound}``: for the lower bound
        * ``{upper_bound}``: for the upper bound

        Args:
            assignment (str): the string containing the assignment template.
        """
        self._assignment = assignment

    def create_assignment(self, parameter_variable, lower_bound, upper_bound):
        assignment = self._assignment.replace('{parameter_variable}', parameter_variable)
        assignment = assignment.replace('{lower_bound}', str(lower_bound))
        assignment = assignment.replace('{upper_bound}', str(upper_bound))
        return assignment


class IdentityTransform(AbstractTransformation):
    """The identity transform does no transformation and returns the input given."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('{parameter_variable}')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('{parameter_variable}')

    def encode_bounds(self, lower_bounds, upper_bounds):
        return lower_bounds, upper_bounds


class PositivityTransform(AbstractTransformation):
    """Restrain the parameter to the positive values, i.e. returns ``max(x, 0)``."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('max({parameter_variable}, (mot_float_type)0)')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('max({parameter_variable}, (mot_float_type)0)')

    def encode_bounds(self, lower_bounds, upper_bounds):
        return np.ones_like(lower_bounds) * -np.inf, upper_bounds


class ClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using the clamp function."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type){parameter_variable}, '
                                           '(mot_float_type){lower_bound}, '
                                           '(mot_float_type){upper_bound})')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type){parameter_variable}, '
                                           '(mot_float_type){lower_bound}, '
                                           '(mot_float_type){upper_bound})')

    def encode_bounds(self, lower_bounds, upper_bounds):
        return np.ones_like(lower_bounds) * -np.inf, np.ones_like(lower_bounds) * np.inf


class ScaleClampTransform(AbstractTransformation):

    def __init__(self, scale):
        """Clamps the value to the given bounds and applies a scaling to bring the parameters in sensible ranges.

        To encode, the parameter value is multiplied by the scaling factor.
        To decode, it is divided by the scaling factor.

        Args:
            scale (float): the scaling factor by which to scale the parameter
        """
        super().__init__()
        self._scale = scale

    def get_cl_encode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type){parameter_variable}, '
                                           '(mot_float_type){lower_bound}, '
                                           '(mot_float_type){upper_bound}) * ' + str(self._scale))

    def get_cl_decode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type){parameter_variable} / ' + str(self._scale) + ', '
                                           '(mot_float_type){lower_bound}, '
                                           '(mot_float_type){upper_bound})')

    def encode_bounds(self, lower_bounds, upper_bounds):
        return np.ones_like(lower_bounds) * -np.inf, np.ones_like(lower_bounds) * np.inf


class ScaleTransform(AbstractTransformation):

    def __init__(self, scale):
        """Applies a scaling to bring the parameters in sensible ranges.

        To encode, the parameter value is multiplied by the scaling factor.
        To decode, it is divided by the scaling factor.

        Args:
            scale (float): the scaling factor by which to scale the parameter
        """
        super().__init__()
        self._scale = scale

    def get_cl_encode(self):
        return FormatAssignmentConstructor('{parameter_variable} * ' + str(self._scale))

    def get_cl_decode(self):
        return FormatAssignmentConstructor('{parameter_variable} / ' + str(self._scale))

    def encode_bounds(self, lower_bounds, upper_bounds):
        return lower_bounds * self._scale, upper_bounds * self._scale


class CosSqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a cos(sqr()) transform."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor(
            'acos(clamp((mot_float_type)sqrt(fabs( ({parameter_variable} - {lower_bound}) / '
            '                                      ({upper_bound} - {lower_bound}) )), '
            '           (mot_float_type)0, (mot_float_type)1))')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('pown(cos({parameter_variable}), 2) * ' +
                                           '({upper_bound} - {lower_bound}) + {lower_bound}')

    def encode_bounds(self, lower_bounds, upper_bounds):
        return np.ones_like(lower_bounds) * -np.inf, np.ones_like(upper_bounds) * np.inf


class SinSqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a sin(sqr()) transform."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor(
            'asin(clamp((mot_float_type)sqrt(fabs( ({parameter_variable} - {lower_bound}) / '
            '                                       ({upper_bound} - {lower_bound}) )), '
            '           (mot_float_type)0, (mot_float_type)1))')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('pown(sin({parameter_variable}), 2) * ' +
                                           '({upper_bound} - {lower_bound}) + {lower_bound}')

    def encode_bounds(self, lower_bounds, upper_bounds):
        return np.ones_like(lower_bounds) * -np.inf, np.ones_like(upper_bounds) * np.inf


class SqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a sqr() transform."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('sqrt({parameter_variable})')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type)({parameter_variable} * {parameter_variable}), '
                                           '      (mot_float_type){lower_bound}, '
                                           '      (mot_float_type){upper_bound})')

    def encode_bounds(self, lower_bounds, upper_bounds):
        return np.ones_like(lower_bounds) * -np.inf, np.ones_like(upper_bounds) * np.inf


class AbsModXTransform(AbstractTransformation):

    def __init__(self, x_cl, x_python):
        """Create an transformation that returns the absolute modulo x value of the input."""
        super().__init__()
        self._x_cl = x_cl
        self._x_python = x_python

    def get_cl_encode(self):
        return FormatAssignmentConstructor(
            '({parameter_variable} - (' + str(self._x_cl) + ' * floor({parameter_variable} / '
            + str(self._x_cl) + ')))')

    def get_cl_decode(self):
        return FormatAssignmentConstructor(
            '({parameter_variable} - (' + str(self._x_cl) + ' * floor({parameter_variable} / '
            + str(self._x_cl) + ')))')

    def encode_bounds(self, lower_bounds, upper_bounds):
        return np.ones_like(lower_bounds) * -np.inf, np.ones_like(upper_bounds) * np.inf


class AbsModPiTransform(AbsModXTransform):
    def __init__(self):
        super().__init__('M_PI', np.pi)


class AbsModTwoPiTransform(AbsModXTransform):
    def __init__(self):
        super().__init__('(2 * M_PI)', 2 * np.pi)
