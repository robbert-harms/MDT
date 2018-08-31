from copy import deepcopy

__author__ = 'Robbert Harms'
__date__ = '2017-07-20'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


# The list of components we are loading at the moment. This allows keeping track of nested components.
_component_loading = []


class ComponentBuilder:
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
        from mdt.lib.components import temporary_component_updates, add_template_component

        cls = self._create_class(template)

        if template.subcomponents:
            subcomponents = template.subcomponents

            class SubComponentConstruct(cls):
                def __init__(self, *args, **kwargs):
                    with temporary_component_updates():
                        for component in subcomponents:
                            add_template_component(component)
                        super().__init__(*args, **kwargs)

            return SubComponentConstruct
        return cls

    def _create_class(self, template):
        """Create a class of the right type given the information in the template.

        This is to be used by subclasses.

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
        class MyGoal:
            def test(self):
                print('test')

        # the template class from which we want to construct a new MyGoal, note the @bind_function
        class MyConfig(ComponentTemplate):
            @bind_function
            def test(self):
                super().test()
                print('test2')

    The component builder takes care to actually bind the new method to the final object.

    What this will do essentially is that it will add the property bind to the function. This should act as a
    flag indicating that that function should be bound.

    Args:
        func (python function): the function to bind to the build object
    """
    func._bind = True
    return func


class ComponentTemplateMeta(type):

    def __new__(mcs, name, bases, attributes):
        """A pre-processor for the components.

        On the moment this meta class does two things, first it adds all functions with the '_bind' property
        to the ``bound_methods`` dictionary for binding them later to the constructed class. Second, it sets the
        ``name`` attribute to the template class name if there is no ``name`` attribute defined.
        """
        result = super().__new__(mcs, name, bases, attributes)

        result.component_type = mcs._resolve_attribute(bases, attributes, '_component_type')
        result.bound_methods = mcs._get_bound_methods(bases, attributes)
        result.subcomponents = mcs._get_subcomponents(attributes)
        result.name = mcs._get_component_name_attribute(name, bases, attributes)

        if len(_component_loading) == 1:
            try:
                if result.component_type is not None and result.name:
                    from mdt.lib.components import add_template_component
                    add_template_component(result)
            except ValueError:
                pass

        _component_loading.pop()
        return result

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        _component_loading.append(name)
        return dict()

    @staticmethod
    def _get_component_name_attribute(name, bases, attributes):
        if 'name' in attributes:
            return attributes['name']
        return name

    @staticmethod
    def _get_bound_methods(bases, attributes):
        """Get all methods in the template that have the ``_bind`` attribute.

        This collects all methods that have the ``_bind`` attribute to a dictionary.

        Returns:
            dict: all the methods that need to be bound.
        """
        bound_methods = {value.__name__: value for value in attributes.values() if hasattr(value, '_bind')}
        for base in bases:
            if hasattr(base, 'bound_methods'):
                for key, value in base.bound_methods.items():
                    if key not in bound_methods:
                        bound_methods.update({key: value})
        return bound_methods

    @staticmethod
    def _get_subcomponents(attributes):
        """Get all the sub-components defined in this template.

        Returns:
            list: the defined sub-components.
        """
        return [value for value in attributes.values() if isinstance(value, ComponentTemplateMeta)]

    @staticmethod
    def _resolve_attribute(bases, attributes, attribute_name, base_predicate=None):
        """Search for the given attribute in the given attributes or in the attributes of the bases.

        Args:
            base_predicate (func): if given, a predicate that runs on the attribute of one of the bases to determine
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


class ComponentTemplate(object, metaclass=ComponentTemplateMeta):
    """The component configuration.

    By overriding the class attributes you can define complex configurations. The actual class distilled from these
    configurations are loaded by the builder referenced by ``_builder``.

    Attributes:
        _component_type (str): the component type of this template. Set to one of the valid template types.
        _builder (ComponentBuilder): the builder to use for constructing an object of the given template
        name (str): the name of the template
        description (str): a description of the object / template
    """
    _component_type = None
    _builder = None
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
        return cls._builder.create_class(cls)

    @classmethod
    def meta_info(cls):
        return {'name': cls.name,
                'description': cls.description,
                'template': deepcopy(cls)}
