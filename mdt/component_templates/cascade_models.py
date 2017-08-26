from copy import deepcopy
import six
import mdt
from mdt.component_templates.base import ComponentBuilder, bind_function, method_binding_meta, ComponentTemplate, \
    register_builder, ComponentTemplateMeta
from mdt.models.cascade import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CascadeTemplateMeta(ComponentTemplateMeta):

    def __new__(mcs, name, bases, attributes):
        name_attribute = ComponentTemplateMeta._resolve_attribute(bases, attributes, 'name')

        result = super(CascadeTemplateMeta, mcs).__new__(mcs, name, bases, attributes)

        if name != 'CascadeTemplate':
            if name_attribute is None:
                if attributes['models']:
                    name_attribute = attributes['models'][-1]

                name_modifier = ComponentTemplateMeta._resolve_attribute(bases, attributes, 'cascade_name_modifier')
                if name_modifier:
                    result.name = '{} (Cascade|{})'.format(name_attribute, name_modifier)
                else:
                    result.name = '{} (Cascade)'.format(name_attribute)
        return result


class CascadeTemplate(six.with_metaclass(CascadeTemplateMeta, ComponentTemplate)):
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

                models = ('BallStick (Cascade)', 'Charmed_r1')

        inits (dict): per model the initializations from the previous model. Example:

            .. code-block:: python

                inits = {'Charmed_r1': [
                            ('Tensor.theta', 'Stick.theta'),
                            ('Tensor.phi', 'Stick.phi'),
                            ('w_res0.w', lambda output_previous, output_all_previous: output_previous['w_stick.w'])
                            ]
                        }

            In this example the Charmed_r1 model in the cascade initializes its Tensor compartment with a previous
            Ball&Stick model and initializes the restricted compartment volume fraction with the Stick fraction.
            You can either provide a string matching the parameter name of the exact previous model, or provide
            callback function that accepts both a dict containing the previous model estimates
            and a list of results from all previous model estimates, your callback function then returns a new
            initialization value (or map) for that parameter.

        fixes (dict): per model the fixations from the previous model. Example:

            .. code-block:: python

                fixes = {'Charmed_r1': [('CharmedRestricted0.theta', 'Stick.theta'),
                                        ('CharmedRestricted0.phi', 'Stick.phi')]}

            The syntax is similar to that of the inits attribute.

        lower_bounds (dict): per model the lower bounds to set using the results from the previous model
            Example:

            .. code-block:: python

                lower_bounds = {'Charmed_r1': [
                    ('S0.s0', lambda output_previous, output_all_previous: 2 * np.min(output_previous['S0.s0']))
                ]}

            The syntax is similar to that of the inits attribute.

        upper_bounds (dict): per model the upper bounds to set using the results from the previous model
            Example:

            .. code-block:: python

                upper_bounds = {'Charmed_r1': [
                    ('S0.s0', lambda output_previous, output_all_previous: 2 * np.max(output_previous['S0.s0']))
                ]}

            The syntax is similar to that of the inits attribute.
    """
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


class CascadeBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class SimpleCascadeModel

        Args:
            template (CascadeTemplate): the cascade config template to use for creating the class with the right init
                settings.
        """
        class AutoCreatedCascadeModel(method_binding_meta(template, SimpleCascadeModel)):

            def __init__(self, *args):
                new_args = [deepcopy(template.name),
                            list(map(mdt.get_model, template.models))]
                for ind, arg in args:
                    new_args[ind] = arg
                super(AutoCreatedCascadeModel, self).__init__(*new_args)

            def _prepare_model(self, iteration_position, model, output_previous, output_all_previous):
                super(AutoCreatedCascadeModel, self)._prepare_model(iteration_position, model,
                                                                    output_previous, output_all_previous)

                def parse_value(v):
                    if isinstance(v, six.string_types):
                        return output_previous[v]
                    elif hasattr(v, '__call__'):
                        return v(output_previous, output_all_previous)
                    return v

                for item in template.inits.get(model.name, {}):
                    model.init(item[0], parse_value(item[1]))
                for item in template.inits.get(iteration_position, {}):
                    model.init(item[0], parse_value(item[1]))

                for item in template.fixes.get(model.name, {}):
                    model.fix(item[0], parse_value(item[1]))
                for item in template.fixes.get(iteration_position, {}):
                    model.fix(item[0], parse_value(item[1]))

                for item in template.lower_bounds.get(model.name, {}):
                    model.set_lower_bound(item[0], parse_value(item[1]))
                for item in template.lower_bounds.get(iteration_position, {}):
                    model.set_lower_bound(item[0], parse_value(item[1]))

                for item in template.upper_bounds.get(model.name, {}):
                    model.set_upper_bound(item[0], parse_value(item[1]))
                for item in template.upper_bounds.get(iteration_position, {}):
                    model.set_upper_bound(item[0], parse_value(item[1]))

                self._prepare_model_cb(iteration_position, model, output_previous, output_all_previous)

        return AutoCreatedCascadeModel


register_builder(CascadeTemplate, CascadeBuilder())
