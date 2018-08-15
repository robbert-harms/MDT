from mdt.models.base import NamedProtocolProblem
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
        super().__init__(*args, **kwargs)

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
        raise NotImplementedError()

    def get_next(self, output_previous_models):
        """Get the next model in the cascade. This is the only function called by the model optimizer.

        This cascade class is supposed to remember which model is next.

        Args:
            output_previous_models (list): The output of all the previous models. Each element are the results
                of the optimization run of one of the models in the cascade.

        Returns:
            DMRIOptimizable: The model used for the next fit
        """
        raise NotImplementedError()

    def reset(self):
        """Reset the iteration over the cascade.

        The implementing class should now reset the iteration such that get_next gets the first model again.
        """
        raise NotImplementedError()

    def get_model(self, name):
        """Get one of the models in the cascade by name.

        Args:
            name (str): the name of the model we want to return

        Returns:
            the model we want to have or None if no model found
        """
        raise NotImplementedError()

    def get_model_names(self):
        """Get the names of the models in this cascade in order of execution.

        Returns:
            list of str: the names of the models in this list
        """
        raise NotImplementedError()

    def set_input_data(self, input_data):
        """Set the input data in every model in the cascade."""
        raise NotImplementedError()

    def update_active_post_processing(self, processing_type, settings):
        """Update the active post-processing semaphores for every model.
        """
        raise NotImplementedError()


class SimpleCascadeModel(DMRICascadeModelInterface):

    def __init__(self, name, model_list):
        """Create a new cascade model from a given list of models.

        This class adds some standard bookkeeping to make implementing cascade models easier.

        Args:
            name (str): the name of this cascade model
            model_list (list of models): the list of models this cascade consists of
        """
        super().__init__()
        self._name = name
        self._model_list = model_list
        self._iteration_position = 0

        model_names = [model.name for model in model_list]
        if len(model_names) > len(set(model_names)):
            raise ValueError('Non-unique model names detected in the cascade. '
                             'Please ensure all models are uniquely named.')

    @property
    def name(self):
        return self._name

    def has_next(self):
        return self._iteration_position != len(self._model_list)

    def get_next(self, output_previous_models):
        next_model = self._model_list[self._iteration_position]

        output_previous = {}
        if self._iteration_position > 0:
            output_previous = output_previous_models[self._iteration_position - 1]

        self._prepare_model(self._iteration_position, next_model, output_previous, output_previous_models)
        self._iteration_position += 1
        return self._set_model_options(next_model)

    def reset(self):
        self._iteration_position = 0

    def is_input_data_sufficient(self, input_data=None):
        return all(model.is_input_data_sufficient(input_data) for model in self._model_list)

    def get_input_data_problems(self, input_data=None):
        problems = []
        for model in self._model_list:
            problems.extend(map(lambda p: NamedProtocolProblem(p, model.name), model.get_input_data_problems(input_data)))
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

    def set_input_data(self, input_data):
        for model in self._model_list:
            model.set_input_data(input_data)

    def update_active_post_processing(self, processing_type, settings):
        for model in self._model_list:
            model.update_active_post_processing(processing_type, settings)

    def _set_model_options(self, model):
        """The final hook before we return a model from this class.

        This can set all kind of additional extra's to the model before we return it using any of the functions
        in this class

        Args:
            model: the model to which we want to set the final functions

        Returns:
            model: the same model with all extra's set.
        """
        return model

    def _prepare_model(self, iteration_position, model, output_previous, output_all_previous):
        """Prepare the next model with the output of the previous model.

        By default this model initializes all parameter maps to the output of the previous model.

        Args:
            iteration_position (int): the index (in the list of cascades) of the model we are initializing.
                First model has position 0, then 1 etc.
            model: The model to prepare
            output_previous (dict): the output of the (direct) previous model.
            output_all_previous (list): The output of all the previous models. Every element (indexed by position in the
                cascade) contains the full set of results from the optimization of that specific model.

        Returns:
            None, preparing should happen in-place.
        """
        if not isinstance(model, DMRICascadeModelInterface) and output_previous:
            for key, value in output_previous.items():
                if model.has_parameter(key):
                    model.init(key, value)


