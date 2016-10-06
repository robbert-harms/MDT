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


class MissingColumns(ModelProtocolProblem):

    def __init__(self, missing_columns):
        super(MissingColumns, self).__init__()
        self.missing_columns = missing_columns

    def __str__(self):
        return 'Missing columns: ' + ', '.join(self.missing_columns)


class InsufficientShells(ModelProtocolProblem):

    def __init__(self, required_nmr_shells, nmr_shells):
        super(InsufficientShells, self).__init__()
        self.required_nmr_shells = required_nmr_shells
        self.nmr_shells = nmr_shells

    def __str__(self):
        return 'Required number of shells is {}, this protocol has {}.'.format(
            self.required_nmr_shells, self.nmr_shells)


class NamedProtocolProblem(ModelProtocolProblem):

    def __init__(self, model_protocol_problem, model_name):
        """This extends the given model protocol problem to also include the name of the model.

        Args:
            model_protocol_problem (ModelProtocolProblem): The name for the problem with the given model.
            model_name (str): the name of the model
        """
        super(NamedProtocolProblem, self).__init__()
        self._model_protocol_problem = model_protocol_problem
        self._model_name = model_name

    def __str__(self):
        return "{0}: {1}".format(self._model_name, self._model_protocol_problem)
