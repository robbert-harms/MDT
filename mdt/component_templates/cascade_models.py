from copy import deepcopy
import mdt
from mdt.component_templates.base import ComponentBuilder, bind_function, ComponentTemplate, ComponentTemplateMeta
from mdt.models.cascade import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CascadeBuilder(ComponentBuilder):

    def _create_class(self, template):
        """Creates classes with as base class SimpleCascadeModel

        Args:
            template (CascadeTemplate): the cascade config template to use for creating the class with the right init
                settings.
        """
        class AutoCreatedCascadeModel(SimpleCascadeModel):

            def __init__(self):
                models = []
                for model_def in template.models:
                    if isinstance(model_def, str):
                        models.append(mdt.get_model(model_def)())
                    else:
                        models.append(mdt.get_model(model_def[0])(model_def[1]))

                super().__init__(deepcopy(template.name), models)

            def _prepare_model(self, iteration_position, model, output_previous, output_all_previous):
                super()._prepare_model(iteration_position, model, output_previous, output_all_previous)

                def parse_value(v):
                    if isinstance(v, str):
                        return output_previous[v]
                    elif hasattr(v, '__call__'):
                        return v(output_previous, output_all_previous)
                    return v

                def apply_func(template_element, cb):
                    items_to_apply = dict(template_element.get(model.name, {}))
                    items_to_apply.update(dict(template_element.get(iteration_position, {})))

                    for key, value in items_to_apply.items():
                        cb(key, parse_value(value))

                apply_func(template.inits, lambda name, value: model.init(name, value))
                apply_func(template.fixes, lambda name, value: model.fix(name, value))
                apply_func(template.lower_bounds, lambda name, value: model.set_lower_bound(name, value))
                apply_func(template.upper_bounds, lambda name, value: model.set_upper_bound(name, value))

                self._prepare_model_cb(iteration_position, model, output_previous, output_all_previous)

        for name, method in template.bound_methods.items():
            setattr(AutoCreatedCascadeModel, name, method)

        return AutoCreatedCascadeModel


class CascadeTemplateMeta(ComponentTemplateMeta):

    @staticmethod
    def _get_component_name_attribute(name, bases, attributes):
        name_attribute = ComponentTemplateMeta._resolve_attribute(bases, attributes, 'name')

        if name != 'CascadeTemplate':
            if attributes.get('models', []):
                if isinstance(attributes['models'][-1], str):
                    name_attribute = attributes['models'][-1]
                else:
                    name_attribute = attributes['models'][-1][1]

            name_modifier = ComponentTemplateMeta._resolve_attribute(bases, attributes, 'cascade_name_modifier')
            if name_modifier:
                name_attribute = '{} (Cascade|{})'.format(name_attribute, name_modifier)
            else:
                name_attribute = '{} (Cascade)'.format(name_attribute)

        return name_attribute


class CascadeTemplate(ComponentTemplate, metaclass=CascadeTemplateMeta):
    """The cascade config to inherit from.

    These configs are loaded on the fly by the CascadeBuilder.

    Attributes:
        name (str): the name of this cascade, if not specified we will name the model to the
            last model in the cascade appended with the cascade type variable.
        cascade_name_modifier (str): the cascade type name modifier. This is used to automatically construct the
            name of the model. The format of the cascade model names is ``<Model> (Cascade|<cascade_name_modifier>)`` if
            some name modifier is given, else it is just ``<Model> (Cascade)``.
        description (str): the description
        models (tuple): the list of models we wish to optimize (in that order) Example:

            .. code-block:: python

                models = ('BallStick_r1 (Cascade)', 'CHARMED_r1')

            It is also possible to rename models, to do so, add a tuple instead of a string. For example:

            .. code-block:: python

                models = (...,
                          ('CHARMED_r1', 'CHARMED_r1_nickname'),
                          ...)

        inits (dict): per model the initializations from the previous model. Example:

            .. code-block:: python

                inits = {'CHARMED_r1': {
                            'Tensor.theta': 'Stick.theta',
                            'Tensor.phi': 'Stick.phi',
                            'w_res0.w': lambda out_previous, out_all_previous: out_all_previous[0]['w_stick.w']
                            }
                        }

            In this example the CHARMED_r1 model in the cascade initializes its Tensor compartment with a previous
            Ball&Stick model and initializes the restricted compartment volume fraction with the Stick fraction.
            You can either provide a string matching the parameter name of the previous model, or provide a
            callback function that accepts both a dict containing the previous model estimates
            and a list of results from all previous model estimates, your callback function then returns a new
            initialization value (or map) for that parameter.

        fixes (dict): per model the fixations from the previous model. Example:

            .. code-block:: python

                fixes = {'CHARMED_r1': {'CharmedRestricted0.theta': 'Stick.theta',
                                        'CharmedRestricted0.phi': 'Stick.phi'}}

            The syntax is similar to that of the inits attribute.

        lower_bounds (dict): per model the lower bounds to set using the results from the previous model
            Example:

            .. code-block:: python

                lower_bounds = {'CHARMED_r1': {
                    'S0.s0': lambda output_previous, output_all_previous: 2 * np.min(output_previous['S0.s0'])
                    }
                }

            The syntax is similar to that of the inits attribute.

        upper_bounds (dict): per model the upper bounds to set using the results from the previous model
            Example:

            .. code-block:: python

                upper_bounds = {'CHARMED_r1': {
                    'S0.s0': lambda output_previous, output_all_previous: 2 * np.max(output_previous['S0.s0'])
                    }
                }

            The syntax is similar to that of the inits attribute.
    """
    _component_type = 'cascade_models'
    _builder = CascadeBuilder()

    name = None
    cascade_name_modifier = ''
    description = ''
    models = ()
    inits = {}
    fixes = {}
    lower_bounds = {}
    upper_bounds = {}

    @bind_function
    def _prepare_model_cb(self, iteration_position, model, output_previous, output_all_previous):
        """Finalize the preparation of the model in this callback.

        This is called at the end of the regular _prepare_model function defined in the SimpleCascadeModel and
        as implemented by the AutoCreatedCascadeModel.

        Use this if you want to control more of the initialization of the next model than only the inits and fixes.

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

    @classmethod
    def meta_info(cls):
        target_model = None
        if cls.models:
            target_model = cls.models[-1]

        cascade_type_name = 'Cascade'
        if cls.cascade_name_modifier:
            cascade_type_name += '|' + cls.cascade_name_modifier

        meta_info = deepcopy(ComponentTemplate.meta_info())
        meta_info.update({'name': cls.name,
                          'description': cls.description,
                          'target_model': target_model,
                          'cascade_type_name': cascade_type_name})
        return meta_info
