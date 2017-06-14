from copy import deepcopy

import six

import mdt
from mdt.components_loader import ComponentConfig, bind_function, get_meta_info, ComponentBuilder, method_binding_meta
from mdt.models.cascade import SimpleCascadeModel

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CascadeConfig(ComponentConfig):
    """The cascade config to inherit from.

    These configs are loaded on the fly by the CascadeBuilder.

    Attributes:
        name (str): the name of this cascade
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
            and a dict containing all previous model estimates by model name and returns a single initialization map
            or value.

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
    name = ''
    description = ''
    models = ()
    inits = {}
    fixes = {}
    lower_bounds = {}
    upper_bounds = {}

    @bind_function
    def _prepare_model_cb(self, model, output_previous, output_all_previous):
        """Finalize the preparation of the model in this callback.

        This is called at the end of the regular _prepare_model function defined in the SimpleCascadeModel and
        as implemented by the AutoCreatedCascadeModel.

        Use this if you want to control more of the initialization of the next model than only the inits and fixes.
        """

    @classmethod
    def meta_info(cls):
        meta_info = deepcopy(ComponentConfig.meta_info())
        meta_info.update({'name': cls.name,
                          'description': cls.description})
        return meta_info


class CascadeBuilder(ComponentBuilder):

    def create_class(self, template):
        """Creates classes with as base class SimpleCascadeModel

        Args:
            template (CascadeConfig): the cascade config template to use for creating the class with the right init
                settings.
        """
        class AutoCreatedCascadeModel(method_binding_meta(template, SimpleCascadeModel)):

            def __init__(self, *args):
                new_args = [deepcopy(template.name),
                            list(map(mdt.get_model, template.models))]
                for ind, arg in args:
                    new_args[ind] = arg
                super(AutoCreatedCascadeModel, self).__init__(*new_args)

            def _prepare_model(self, model, output_previous, output_all_previous):
                super(AutoCreatedCascadeModel, self)._prepare_model(model, output_previous, output_all_previous)

                def parse_value(v):
                    if isinstance(v, six.string_types):
                        return output_previous[v]
                    elif hasattr(v, '__call__'):
                        return v(output_previous, output_all_previous)
                    return v

                for item in template.inits.get(model.name, {}):
                    model.init(item[0], parse_value(item[1]))

                for item in template.fixes.get(model.name, {}):
                    model.fix(item[0], parse_value(item[1]))

                for item in template.lower_bounds.get(model.name, {}):
                    model.set_lower_bound(item[0], parse_value(item[1]))

                for item in template.upper_bounds.get(model.name, {}):
                    model.set_upper_bound(item[0], parse_value(item[1]))

                self._prepare_model_cb(model, output_previous, output_all_previous)

        return AutoCreatedCascadeModel
