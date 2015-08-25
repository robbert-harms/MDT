__author__ = 'Robbert Harms'
__date__ = "2015-08-25"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelProtocolProblem(object):

    def __init__(self):
        """The base class for indicating problems with a protocol.

        These are meant to be returned from the function get_protocol_problems() from the ProtocolCheckInterface.

        Each of these problems is supposed to overwrite the function __str__() for reporting the problem.
        """

    def __repr__(self):
        return self.__str__()

    def can_merge(self, other_problem):
        """If this problem object can merge with the other problem object.

        This can for example always return False if this object can not merge at all. Or can say True to anything
        for merging anything.

        In general it will return True if the problems are of the same class.

        Args:
            other_problem (ModelProtocolProblem): The protocol problem to merge with this one.

        Returns:
            boolean: True if this problem can merge with the other_problem, false otherwise.
        """

    def merge(self, other_problem):
        """Merge another model protocol problem of the same kind into one problem.

        Args:
            other_problem (ModelProtocolProblem): The protocol problem to merge with this one.

        Returns:
            ModelProtocolProblem: A new protocol problem with merged information.
        """


class MissingColumns(ModelProtocolProblem):

    def __init__(self, missing_columns):
        super(MissingColumns, self).__init__()
        self.missing_columns = missing_columns

    def __str__(self):
        return 'Missing columns: ' + ', '.join(self.missing_columns)

    def can_merge(self, other_problem):
        return isinstance(other_problem, MissingColumns)

    def merge(self, other_problem):
        return MissingColumns(self.missing_columns + other_problem.missing_columns)


class InsufficientShells(ModelProtocolProblem):

    def __init__(self, required_nmr_shells, nmr_shells):
        super(InsufficientShells, self).__init__()
        self.required_nmr_shells = required_nmr_shells
        self.nmr_shells = nmr_shells

    def __str__(self):
        return 'Required number of shells is {}, this protocol has {}.'.format(
            self.required_nmr_shells, self.nmr_shells)

    def can_merge(self, other_problem):
        return isinstance(other_problem, InsufficientShells)

    def merge(self, other_problem):
        return InsufficientShells(self.nmr_shells, max(self.required_nmr_shells, other_problem.required_nmr_shells))


class NamedProtocolProblem(ModelProtocolProblem):

    def __init__(self, model_protocol_problem, model_name):
        """This extends the model protocol problem to also include the name of the model.

        Args:
            model_protocol_problem (ModelProtocolProblem): The name for the problem with the given model.
            model_name (str): the name of the model
        """
        super(NamedProtocolProblem, self).__init__()
        self._model_protocol_problem = model_protocol_problem
        self._model_name = model_name

    def __str__(self):
        return "{0}: {1}".format(self._model_name, self._model_protocol_problem)

    def can_merge(self, other_problem):
        return False

    def merge(self, other_problem):
        raise ValueError("This class does not support merging.")