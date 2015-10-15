import mdt
from mdt.utils import simple_parameter_init, ProtocolCheckInterface, condense_protocol_problems

__author__ = 'Robbert Harms'
__date__ = "2015-04-24"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CascadeModelInterface(object):

    def __init__(self):
        """The interface to cascade models.

        A cascade model is a model consisting of multi-compartment models or other cascade models. The idea is that
        it contains a number of models that are to be ran one after each other and with which the output results of
        the previous fit_model(s) are used for the next fit_model.
        """
        self._double_precision = False

    @property
    def name(self):
        """Get the name of this cascade model.

        Returns:
            str: The name of this cascade model
        """
        return ''

    @property
    def double_precision(self):
        return self._double_precision

    @double_precision.setter
    def double_precision(self, double_precision):
        self._double_precision = double_precision
        self._set_double_precision(double_precision)

    def has_next(self):
        """Check if this cascade model has a next model.

        Returns:
            boolean: True if there is a next model, false otherwise.
        """

    def get_next(self, output_previous_models):
        """Get the next model in the cascade. This is the only function called by the cascade model optimizer

        This class is supposed to remember which model it gave the optimizer in what order.

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

    def is_protocol_sufficient(self, protocol):
        """Check if the given protocol holds enough information for all models in the cascade.

        Args:
            protocol (Protocol): The protocol object to check for sufficient information.

        Returns:
            boolean: True if there is enough information in the protocol, false otherwise
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

    def _set_double_precision(self, double_precision):
        """Set the value double precision for all models in the cascade.

        Args:
            double_precision (boolean): the value to set for all models in the cascade
        """

    def set_problem_data(self, problem_data):
        """Set the problem data in every model in the cascade."""

    def set_gradient_deviations(self, grad_dev):
        """Set the gradient deviations in every model."""


class SimpleCascadeModel(CascadeModelInterface, ProtocolCheckInterface):

    def __init__(self, name, model_list):
        """Create a new cascade model from a given list of models.

        This class adds some standard bookkeeping to make implementing cascade models easier.

        Args:
            name (str): the name of this cascade model
            model_list (list of models): the list of models this cascade consists of
        """
        super(CascadeModelInterface, self).__init__()
        self._name = name
        self._model_list = model_list
        self._iteration_position = 0

    @property
    def name(self):
        return self._name

    def has_next(self):
        return self._iteration_position != len(self._model_list)

    def get_next(self, output_previous_models):
        next_model = self._model_list[self._iteration_position]
        output_previous = {}
        if self._iteration_position > 0:
            output_previous = output_previous_models[self._model_list[self._iteration_position - 1].name]
        self._prepare_model(next_model, output_previous, output_previous_models)
        self._iteration_position += 1
        return next_model

    def reset(self):
        self._iteration_position = 0

    def is_protocol_sufficient(self, protocol=None):
        for model in self._model_list:
            if not model.is_protocol_sufficient(protocol):
                return False
        return True

    def get_protocol_problems(self, protocol=None):
        return condense_protocol_problems([model.get_protocol_problems(protocol) for model in self._model_list])

    def get_required_protocol_names(self):
        protocol_names = []
        for model in self._model_list:
            protocol_names.extend(model.get_required_protocol_names())
        return list(set(protocol_names))

    def get_model(self, name):
        for model in self._model_list:
            if model.name == name:
                return model
        return None

    def get_model_names(self):
        return [model.name for model in self._model_list]

    def set_problem_data(self, problem_data):
        for model in self._model_list:
            model.set_problem_data(problem_data)

    def set_gradient_deviations(self, grad_dev):
        for model in self._model_list:
            model.set_gradient_deviations(grad_dev)

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
        if not isinstance(model, CascadeModelInterface):
            simple_parameter_init(model, output_previous)

    def _set_double_precision(self, double_precision):
        for model in self._model_list:
            model.double_precision = double_precision


def cascade_builder_decorator(original_class):
    """This function can be used as a decorator for building cascade models.

    By using this decorator you can almost declaratively construct a cascade model.

    This decorator will overwrite the init method to include the name and the models (it will create the models
    from a string name). It also adds the static method get_meta_data() to get the meta data for the components list.

    The actual preferred way of building cascade models is by using the CascadaBuilderMetaClass and by inheriting
    from the CascadeModelBuilder. If however you have a strong need of using a decorator you can use this one.

    Example usage:
        @cascade_builder_decorator
        class BallStick(SimpleCascadeModel):
            name = 'BallStick (Cascade)'
            description = 'Cascade for Ballstick'
            models = ('s0', 'BallStick')

    Args:
        original_class (class): the class we want to wrap
    """
    orig_init = original_class.__init__

    def __init__(self, *args, **kws):
        if len(args) == 2:
            # inheritance is used, the name and model list are already set
            orig_init(self, *args, **kws)
        else:
            orig_init(self, original_class.name, list(map(mdt.get_model, original_class.models)), **kws)

    def get_meta_data():
        return {'name': original_class.name,
                'model_constructor': original_class,
                'description': original_class.description}

    original_class.__init__ = __init__
    original_class.get_meta_data = staticmethod(get_meta_data)

    return original_class


class CascadeBuilderMetaClass(type):

    def __new__(mcs, name, bases, dct):
        """Adds methods to the class at class creation time."""
        result_class = super(CascadeBuilderMetaClass, mcs).__new__(mcs, name, bases, dct)

        def get_meta_data():
            return {'name': result_class.name,
                    'model_constructor': result_class,
                    'description': result_class.description}

        result_class.get_meta_data = staticmethod(get_meta_data)

        orig_init = result_class.__init__

        def __init__(self, *args, **kws):
            if len(args) == 2:
                # inheritance is used, the name and model list are already set
                orig_init(self, *args, **kws)
            else:
                orig_init(self, result_class.name, list(map(mdt.get_model, result_class.models)), **kws)

        result_class.__init__ = __init__

        return result_class


class CascadeModelBuilder(SimpleCascadeModel, metaclass=CascadeBuilderMetaClass):
    """The model builder to inherit from.

    One can use this to create models in a declarative style. Example of such a model definition:

    class BallStick(CascadeModelBuilder):
        name = 'BallStick (Cascade)'
        description = 'Cascade for Ballstick'
        models = ('s0', 'BallStick')

    This class has a metaclass which is able to use the class variables to guide the construction of the model.
    """
    __metaclass__ = CascadeBuilderMetaClass
