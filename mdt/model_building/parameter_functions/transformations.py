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


class PositivityTransform(AbstractTransformation):
    """Restrain the parameter to the positive values, i.e. returns ``max(x, 0)``."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('max({parameter_variable}, (mot_float_type)0)')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('max({parameter_variable}, (mot_float_type)0)')


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


class ScaleClampTransform(AbstractTransformation):

    def __init__(self, scale):
        """Clamps the value to the given bounds and applies a scaling to bring the parameters in sensible ranges.

        The given scaling factor should be without the scaling factor. To encode, the parameter value is multiplied
        by the scaling factor. To decode, it is divided by the scaling factor.

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


class SqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a sqr() transform."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('sqrt({parameter_variable})')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type)({parameter_variable} * {parameter_variable}), '
                                           '      (mot_float_type){lower_bound}, '
                                           '      (mot_float_type){upper_bound})')


class AbsModXTransform(AbstractTransformation):

    def __init__(self, x):
        """Create an transformation that returns the absolute modulo x value of the input."""
        super().__init__()
        self._x = x

    def get_cl_encode(self):
        return FormatAssignmentConstructor(
            '({parameter_variable} - (' + str(self._x) + ' * floor({parameter_variable} / ' + str(self._x) + ')))')

    def get_cl_decode(self):
        return FormatAssignmentConstructor(
            '({parameter_variable} - (' + str(self._x) + ' * floor({parameter_variable} / ' + str(self._x) + ')))')


class AbsModPiTransform(AbsModXTransform):
    def __init__(self):
        super().__init__('M_PI')


class AbsModTwoPiTransform(AbsModXTransform):
    def __init__(self):
        super().__init__('(2 * M_PI)')

