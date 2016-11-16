__author__ = 'Robbert Harms'
__date__ = "2015-10-27"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRIOptimizable(object):

    def __init__(self, *args, **kwargs):
        """This is an interface for some base methods we expect in an MRI model.

        Since we have both composite dMRI models and cascade models we must have an overarching interface to make
        sure that both type of models implement the same additional methods.

        The methods in this interface have little to do with modelling, but unify some extra required methods
        in the cascades and composite models.

        Attributes:
            problems_to_analyze (list): the list with problems we want to analyze. Suppose we have a few thousands
                problems defined in this model, but we want to run the optimization only on a few problems. By setting
                this attribute to a list of problems indices only those problems will be analyzed.
            double_precision (boolean): if we do the computations in double or float precision
        """
        super(DMRIOptimizable, self).__init__()
        self.problems_to_analyze = None
        self.double_precision = False

    def is_protocol_sufficient(self, protocol=None):
        """Check if the protocol holds enough information for this model to work.

        Args:
            protocol (Protocol): The protocol object to check for sufficient information. If set the None, the
                current protocol in the problem data is used.

        Returns:
            boolean: True if there is enough information in the protocol, false otherwise
        """

    def get_protocol_problems(self, protocol=None):
        """Get all the problems with the protocol.

        Args:
            protocol (Protocol): The protocol object to check for problems. If set the None, the
                current protocol in the problem data is used.

        Returns:
            list of ModelProtocolProblem: A list of :class:`~mdt.model_protocol_problem.ModelProtocolProblem` instances
                or subclasses of that baseclass. These objects indicate the problems with the protocol and this model.
        """

    def get_required_protocol_names(self):
        """Get a list with the constant data names that are needed for this model to work.

        For example, an implementing diffusion MRI model might require the presence of the protocol parameter
        ``g`` and ``b``. This function should then return ``('g', 'b')``.

        Returns:
            :class:`list`: A list of columns names that are to be taken from the protocol data.
        """
