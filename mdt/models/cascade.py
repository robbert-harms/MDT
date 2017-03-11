from mdt.model_protocol_problem import NamedProtocolProblem
from mdt.models.base import DMRIOptimizable

__author__ = 'Robbert Harms'
__date__ = "2015-04-24"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICascadeModelInterface(DMRIOptimizable):

    def __init__(self, *args, **kwargs):
        """The interface to cascade models.

        A cascade model is a model consisting of multi-compartment models or other cascade models. The idea is that
        it contains a number of models that are to be optimized one after another with the output results of
        the previous fit used to initialize the next model.
        """
        super(DMRICascadeModelInterface, self).__init__(*args, **kwargs)
        self.double_precision = False

    @property
    def name(self):
        """Get the name of this cascade model.

        Returns:
            str: The name of this cascade model
        """
        return ''

    def has_next(self):
        """Check if this cascade model has a next model.

        Returns:
            boolean: True if there is a next model, false otherwise.
        """

    def get_next(self, output_previous_models):
        """Get the next model in the cascade. This is the only function called by the model optimizer.

        This cascade class is supposed to remember which model is next.

        Args:
            output_previous_models (dict): The output of all the previous models. The first level of the
                dict is for the models and is indexed by the model name. The second layer contains all the maps.

        Returns:
            SampleModelInterface: The sample model used for the next fit_model.
        """

    def reset(self):
        """Reset the iteration over the cascade.

        The implementing class should now reset the iteration such that get_next gets the first model again.
        """

    def get_model(self, name):
        """Get one of the models in the cascade by name.

        Args:
            name (str): the name of the model we want to return

        Returns:
            the model we want to have or None if no model found
        """

    def get_model_names(self):
        """Get the names of the models in this cascade in order of execution.

        Returns:
            list of str: the names of the models in this list
        """

    def set_problem_data(self, problem_data):
        """Set the problem data in every model in the cascade."""


class SimpleCascadeModel(DMRICascadeModelInterface):

    def __init__(self, name, model_list):
        """Create a new cascade model from a given list of models.

        This class adds some standard bookkeeping to make implementing cascade models easier.

        Args:
            name (str): the name of this cascade model
            model_list (list of models): the list of models this cascade consists of
        """
        super(DMRICascadeModelInterface, self).__init__()
        self._name = name
        self._model_list = model_list
        self._iteration_position = 0
        self.problems_to_analyze = None

    @property
    def name(self):
        return self._name

    def has_next(self):
        return self._iteration_position != len(self._model_list)

    def get_next(self, output_previous_models):
        next_model = self._model_list[self._iteration_position]

        output_previous = {}

        if self._iteration_position > 0:
            previous_model = self._model_list[self._iteration_position - 1]
            output_previous = output_previous_models[previous_model.name]

        self._prepare_model(next_model, output_previous, output_previous_models)
        self._iteration_position += 1
        return self._set_model_options(next_model)

    def reset(self):
        self._iteration_position = 0

    def is_protocol_sufficient(self, protocol=None):
        return all(model.is_protocol_sufficient(protocol) for model in self._model_list)

    def get_protocol_problems(self, protocol=None):
        problems = []
        for model in self._model_list:
            problems.extend(map(lambda p: NamedProtocolProblem(p, model.name), model.get_protocol_problems(protocol)))
        return problems

    def get_required_protocol_names(self):
        protocol_names = []
        for model in self._model_list:
            protocol_names.extend(model.get_required_protocol_names())
        return list(set(protocol_names))

    def get_model(self, name):
        for model in self._model_list:
            if model.name == name:
                return self._set_model_options(model)
        return None

    def get_model_names(self):
        return [model.name for model in self._model_list]

    def set_problem_data(self, problem_data):
        for model in self._model_list:
            model.set_problem_data(problem_data)

    def _set_model_options(self, model):
        """The final hook before we return a model from this class.

        This can set all kind of additional extra's to the model before we return it using any of the functions
        in this class

        Args:
            model: the model to which we want to set the final functions

        Returns:
            model: the same model with all extra's set.
        """
        if self.problems_to_analyze:
            model.problems_to_analyze = self.problems_to_analyze
        model.double_precision = self.double_precision
        return model

    def _prepare_model(self, model, output_previous, output_all_previous):
        """Prepare the next model with the output of the previous model.

        By default this model initializes all parameter maps to the output of the previous model.

        Args:
            model: The model to prepare
            output_previous (dict): the output of the (direct) previous model.
            output_all_previous (dict): The output of all the previous models. Indexed first by model name, second
                by full parameter name.

        Returns:
            None, preparing should happen in-place.
        """
        if not isinstance(model, DMRICascadeModelInterface) and output_previous:
            for key, value in output_previous.items():
                if model.has_parameter(key):
                    model.init(key, value)


