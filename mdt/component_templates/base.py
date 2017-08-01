from copy import deepcopy
from six import with_metaclass

__author__ = 'Robbert Harms'
__date__ = '2017-07-20'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


"""Contains a lookup table that links template types to builders."""
_builder_lookup = {}


class ComponentBuilder(object):
    """The base class for component builders.

    Component builders, together with ComponentTemplate allow you to define components using Templates,
    special classes where the properties are defined using class attributes.

    The ComponentTemplate contains class attributes defining the component which can be used by the
    ComponentBuilder to create a class of the right type from the information in that template.
    """

    def create_class(self, template):
        """Create a class of the right type given the information in the template.

        Args:
            template (ComponentTemplate): the information as a component config

        Returns:
            class: the class of the right type
        """
        raise NotImplementedError()


def bind_function(func):
    """This decorator is for methods in ComponentTemplates that we would like to bind to the constructed component.

    Example suppose you want to inherit or overwrite a function in the constructed model, then in your template/config
    you should define the function and add @bind_function to it as a decorator, like this:

    .. code-block:: python

        # the class we want to create
        class MyGoal(object):
            def test(self):
                print('test')

        # the template class from which we want to construct a new MyGoal, note the @bind_function
        class MyConfig(ComponentTemplate):
            @bind_function
            def test(self):
                super(MyGoal, self).test()
                print('test2')

    The component builder takes care to actually bind the new method to the final object.

    What this will do essentially is that it will add the property bind to the function. This should act as a
    flag indicating that that function should be bound.

    Args:
        func (python function): the function to bind to the build object
    """
    func._bind = True
    return func


def method_binding_meta(template, *bases):
    """Adds all bound functions from the ComponentTemplate to the class being constructed.

     This returns a metaclass similar to the with_metaclass of the six library.

     Args:
         template (ComponentTemplate): the component config with the bound_methods attribute which we will all add
            to the attributes of the to creating class.
     """
    class ApplyMethodBinding(type):
        def __new__(mcs, name, bases, attributes):
            attributes.update(template.bound_methods)
            return super(ApplyMethodBinding, mcs).__new__(mcs, name, bases, attributes)

    return with_metaclass(ApplyMethodBinding, *bases)


class ComponentTemplateMeta(type):

    def __new__(mcs, name, bases, attributes):
        """A pre-processor for the components.

        On the moment this meta class does two things, first it adds all functions with the '_bind' property
        to the bound_methods list for binding them later to the constructed class. Second, it sets the 'name' attribute
        of the component to the class name if there is no name attribute defined.
        """
        result = super(ComponentTemplateMeta, mcs).__new__(mcs, name, bases, attributes)
        bound_methods = {value.__name__: value for value in attributes.values() if hasattr(value, '_bind')}
        for base in bases:
            if hasattr(base, 'bound_methods'):
                for key, value in base.bound_methods.items():
                    if key not in bound_methods:
                        bound_methods.update({key: value})
        result.bound_methods = bound_methods

        if 'name' not in attributes:
            result.name = name

        return result

    @staticmethod
    def _resolve_attribute(bases, attributes, attribute_name, base_predicate=None):
        """Search for the given attribute in the given attributes or in the attributes of the bases.

        Args:
            base_predicate (func): if given a predicate that runs on the attribute of one of the bases to determine
                if we will return that one.

        Returns:
            The value for the attribute

        Raises:
            ValueError: if the attribute could not be found in the attribute or any of the bases
        """
        base_predicate = base_predicate or (lambda _: True)

        if attribute_name in attributes:
            return attributes[attribute_name]
        for base in bases:
            if hasattr(base, attribute_name) and base_predicate(getattr(base, attribute_name)):
                return getattr(base, attribute_name)
        raise ValueError('Attribute not found in this component config or its superclasses.')


class ComponentTemplate(with_metaclass(ComponentTemplateMeta, object)):
    """The component configuration.

    By overriding the class attributes you can define complex configurations. The actual class distilled from these
    configurations are loaded by the ComponentBuilder
    """
    name = ''
    description = ''

    def __new__(cls, *args, **kwargs):
        """Instead of creating an instance of a Template, this will build the actual component.

        This allows one to build the model of a template by regular object initialization. For example, these two
        calls (a and b) are exactly the same::

            template = Template
            a = construct_component(template)
            b = template()
        """
        return construct_component(cls)()

    @classmethod
    def meta_info(cls):
        return {'name': cls.name,
                'description': cls.description,
                'template': deepcopy(cls)}


def construct_component(template):
    """Construct the component from configuration to derived class.

    This function will perform the lookup from config type to builder class and will subsequently build the
    component class from the config.

    Args:
        template (Type[mdt.component_templates.base.ComponentTemplate]): the template we wish to construct into a class.

    Returns:
        class: the constructed class from the given component
    """
    for template_type, builder in _builder_lookup.items():
        if issubclass(template, template_type):
            return builder.create_class(template)
    raise ValueError("No suitable builder for the given component could be found.")


def register_builder(template_type, builder):
    """Register the use of the given builder for creating components from the given template.

    Args:
        template_type (Type[ComponentTemplate]): the template we are assigning
        builder (ComponentBuilder): the builder that can build the given template
    """
    _builder_lookup[template_type] = builder
