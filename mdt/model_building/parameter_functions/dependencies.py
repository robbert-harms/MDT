__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterDependency:

    @property
    def pre_transform_code(self):
        """Some code that may be prefixed to this parameter dependency.

        Here one can put more elaborate CL code. Please make sure that additional variables are unique.

        Returns:
            str: The pre transformation code. This is prepended to the dependency function.
        """
        return ''

    @property
    def assignment_code(self):
        """Get the assignment code (including a ;).

        Returns:
            str: The assignment code.
        """
        return ''


class SimpleAssignment(AbstractParameterDependency):

    def __init__(self, assignment_code):
        """Adds a simple parameter dependency rule for the given parameter.

        This is for one parameter, a simple one-line transformation dependency.

        Args:
            assignment_code (str): the assignment code (in CL) for this parameter
        """
        self._assignment = assignment_code

    @property
    def assignment_code(self):
        return self._assignment
