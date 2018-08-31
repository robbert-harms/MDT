__author__ = 'Robbert Harms'
__date__ = "2015-10-27"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRIOptimizable:

    def __init__(self, *args, **kwargs):
        """This is an interface for some base methods we expect in an MRI model.

        Since we have both composite dMRI models and cascade models we must have an overarching interface to make
        sure that both type of models implement the same additional methods.

        The methods in this interface have little to do with modelling, but unify some extra required methods
        in the cascades and composite models.
        """
        super().__init__()

    def is_input_data_sufficient(self, input_data=None):
        """Check if the input data has enough information for this model to work.

        Args:
            input_data (mdt.utils.MRIInputData): The input data we intend on using with this model.

        Returns:
            boolean: True if there is enough information in the input data, false otherwise.
        """

    def get_input_data_problems(self, input_data=None):
        """Get all the problems with the protocol.

        Args:
            input_data (mdt.utils.MRIInputData): The input data we intend on using with this model.

        Returns:
            list of InputDataProblem: A list of
                InputDataProblem instances or subclasses of that baseclass.
                These objects indicate the problems with the protocol and this model.
        """

    def get_required_protocol_names(self):
        """Get a list with the constant data names that are needed for this model to work.

        For example, an implementing diffusion MRI model might require the presence of the protocol parameter
        ``g`` and ``b``. This function should then return ``('g', 'b')``.

        Returns:
            :class:`list`: A list of columns names that are to be taken from the protocol data.
        """

    def update_active_post_processing(self, processing_type, settings):
        """Update the active post-processing semaphores.

        It is possible to control which post-processing routines get run by overwriting them using this method.
        For a list of post-processors, please see the default mdt configuration file under ``active_post_processing``.

        Args:
            processing_type (str): one of ``sample`` or ``optimization``.
            settings (dict): the items to set in the post-processing information
        """
        raise NotImplementedError()


class InputDataProblem:

    def __init__(self):
        """The base class for indicating problems with the input data.

        These are meant to be returned from the function get_input_data_problems().

        Each of these problems is supposed to overwrite the function __str__() for reporting the problem.
        """

    def __repr__(self):
        return self.__str__()


class MissingProtocolInput(InputDataProblem):

    def __init__(self, missing_columns):
        super().__init__()
        self.missing_columns = missing_columns

    def __str__(self):
        return 'Missing columns: ' + ', '.join(self.missing_columns)


class NamedProtocolProblem(InputDataProblem):

    def __init__(self, model_protocol_problem, model_name):
        """This extends the given model protocol problem to also include the name of the model.

        Args:
            model_protocol_problem (InputDataProblem): The name for the problem with the given model.
            model_name (str): the name of the model
        """
        super().__init__()
        self._model_protocol_problem = model_protocol_problem
        self._model_name = model_name

    def __str__(self):
        return "{0}: {1}".format(self._model_name, self._model_protocol_problem)
