"""The general components loader.

This modules consists of two main items, component sources and component loaders. The component loaders have a list
of sources from which they load the available components.
"""
import glob
import inspect
import os
import imp #todo in P3.4 replace imp calls with importlib.SourceFileLoader(name, path).load_module(name)
from six import with_metaclass
from mot.model_building.cl_functions.base import ModelFunction, LibraryFunction
import mot.model_building.cl_functions.library_functions
import mot.model_building.cl_functions.model_functions

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_model(model_name, **kwargs):
    """Load one of the available models.

    Args:
        model_name (str): One of the models from get_list_of_single_models() or get_list_of_cascade_models()
        **kwargs: Extra keyword arguments used for the initialization of the model

    Returns:
        Either a cascade model or a single model. In any case, a model that can be given to the fit_model function.
    """
    cml = CascadeModelsLoader()
    sml = SingleModelsLoader()
    try:
        return cml.load(model_name, **kwargs)
    except ImportError:
        try:
            return sml.load(model_name, **kwargs)
        except ImportError:
            raise ValueError('The model with the name "{}" could not be found.'.format(model_name))


class ComponentBuilder(object):

    def __init__(self):
        """The base class for component builders.

        Component builders, together with ComponentConfig allow you to define components using class attributes.

        The idea is that the ComponentConfig contains class attributes defining the component and that the
        ComponentBuilder is able to create a class of the right type from the information in the component config.
        """

    def create_class(self, template):
        """Create a class of the right type given the information in the template.

        Args:
            template (ComponentConfig): the information as a component config

        Returns:
            class: the class of the right type
        """


def bind_function(func):
    """This decorator is for methods in ComponentConfigs that we would like to bind to the constructed component.

    Example suppose you want to inherit or overwrite a function in the constructed model, then in your template/config
    you should define the function and add @bind_function to it as a decorator, like this:

    .. code-block:: python

        # the class we want to create
        class MyGoal(object):
            def test(self):
                print('test')

        # the template class from which we want to construct a new MyGoal, note the @bind_function
        class MyConfig(ComponentConfig):
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
    """Adds all bound functions from the ComponentConfig to the class being constructed.

     This returns a metaclass similar to the with_metaclass of the six library.

     Args:
         template (ComponentConfig): the component config with the bound_methods attribute which we will all add
            to the attributes of the to creating class.
     """
    class ApplyMethodBinding(type):
        def __new__(mcs, name, bases, attributes):
            attributes.update(template.bound_methods)
            return super(ApplyMethodBinding, mcs).__new__(mcs, name, bases, attributes)

    return with_metaclass(ApplyMethodBinding, *bases)


class ComponentConfigMeta(type):

    def __new__(mcs, name, bases, attributes):
        """A pre-processor for the components.

        On the moment this meta class does two things, first it adds all functions with the '_bind' property
        to the bound_methods list for binding them later to the constructed class. Second, it sets the 'name' attribute
        of the component to the class name if there is no name attribute defined.
        """
        result = super(ComponentConfigMeta, mcs).__new__(mcs, name, bases, attributes)
        bound_methods = {value.__name__: value for value in attributes.values() if hasattr(value, '_bind')}
        for base in bases:
            if hasattr(base, 'bound_methods'):
                bound_methods.update(base.bound_methods)
        result.bound_methods = bound_methods

        if 'name' not in attributes:
            result.name = name

        return result


class ComponentConfig(with_metaclass(ComponentConfigMeta, object)):
    """The component configuration.

    By overriding the class attributes you can define complex configurations. The actual class distilled from these
    configurations are loaded by the ComponentBuilder
    """
    name = ''
    description = ''

    @classmethod
    def meta_info(cls):
        return {'name': cls.name,
                'description': cls.description}


class ComponentsLoader(object):

    def __init__(self, sources):
        """The base class for loading and displaying components.

        Args:
            sources (:class:`list`): the list of sources to use for loading the components
        """
        self._sources = sources

    def list_all(self):
        """List the names of all the available components."""
        components = []
        for source in self._sources:
            components.extend(source.list())
        return list(sorted(set(components)))

    def has_component(self, name):
        """Check if this loader has a component with the given name.

        Args:
            name (str): the name of the component

        Returns:
            boolean: true if this loader has the given component, false otherwise
        """
        try:
            self._get_preferred_source(name)
            return True
        except ImportError:
            return False

    def get_all_meta_info(self):
        """Get the meta information of all loadable components.

        Returns:
            dict: the keys are the names of the objects, as returned by list_all() and the values are the
                meta-information dicts.
        """
        return {k: self.get_meta_info(k) for k in self.list_all()}

    def get_meta_info(self, name):
        """Get the meta information of a component of the given name.

        Args:
            name (str): The name of the component we want to use

        Returns:
            dict: a dictionary with meta information for this component. Standard meta information is:
                - name (str): the name of the component
                - description (str): the description of the component
        """
        source = self._get_preferred_source(name)
        return source.get_meta_info(name)

    def get_class(self, name):
        """Get the class to the component of the given name.

        Args:
            name (str): The name of the component we want to use

        Returns:
            class or cb function to construct the given component
        """
        source = self._get_preferred_source(name)
        return source.get_class(name)

    def load(self, name, *args, **kwargs):
        """Load the component with the given name

        Args:
            name (str): The name of the component we want to use
            *args: passed to the component
            **kwargs: passed to the component

        Returns:
            the loaded module
        """
        c = self.get_class(name)
        return c(*args, **kwargs)

    def _get_preferred_source(self, name):
        """Try to get the preferred source for the component with the given name.

        The order of the sources matter, the first source takes precedence over the latter ones and so forth.
        """
        for source in self._sources:
            try:
                source.get_class(name)
                return source
            except ImportError:
                pass
        raise ImportError("No component found with the name {}".format(name))


class ComponentsSource(object):

    def __init__(self):
        """Defines a source for components.

        This has functions for listing the available components as well as getting the class and meta information.
        """

    def list(self):
        """Get the names of all the available components from this source.

        Returns:
            list or str: list of the names of all the components loadable from this source.
        """
        return []

    def get_class(self, name):
        """Get the class for the component by the given name

        Args:
            name (str): The name of the component we want to use

        Returns:
            the construction function
        """
        raise ImportError

    def get_meta_info(self, name):
        """Get the meta information of a component of the given name.

        Args:
            name (str): The name of the component we want to use

        Returns:
            dict: a dictionary with meta information for this component. Standard meta information is:
                - name (str): the name of the component
                - description (str): the description of the component
        """
        return {}


class UserComponentsSourceSingle(ComponentsSource):

    def __init__(self, user_type, component_type):
        """

        This expects that the available python files contain a class with the same name as the file in which the class
        resides. For example, if we have a python file named "MySingleComponent.py" we should at least have the class
        named "MySingleComponent" in that python file. Additionally, the file can contain a dictionary named
        'meta_info' which contains meta information about that module.

        Args:
            user_type (str): either 'standard' or 'user'
            component_type (str): the type of component we wish to use. This should be named exactly to one of the
                directories available in mdt/data/components/
        """
        super(UserComponentsSourceSingle, self).__init__()
        self.path = _get_components_path(user_type, component_type)

    def list(self):
        if os.path.isdir(self.path):
            items = []
            for item in os.listdir(self.path):
                if item.endswith('.py') and item != '__init__.py':
                    items.append(item[0:-3])
            return items
        return []

    def get_class(self, name):
        path = os.path.join(self.path, name + '.py')
        if os.path.exists(path):
            module = imp.load_source(name, path)
            return getattr(module, name)
        raise ImportError

    def get_meta_info(self, name):
        path = os.path.join(self.path, name + '.py')
        if os.path.exists(path):
            module = imp.load_source(name, path)
            try:
                return getattr(module, 'meta_info')
            except AttributeError:
                try:
                    cls = self.get_class(name)
                    return cls.meta_info()
                except AttributeError:
                    return {}
        return {}


class AutoUserComponentsSourceSingle(UserComponentsSourceSingle):

    def __init__(self, user_type, component_type, component_builder):
        """

        This class extends the default single components source loader by also being able to use components defined
        using the ComponentConfig method. This means that the components are defined as subclasses of ComponentConfig
        and we need a ComponentBuilder to actually create the components.

        Args:
            user_type (str): either 'standard' or 'user'
            component_type (str): the type of component we wish to use. This should be named exactly to one of the
                directories available in mdt/data/components/
            component_builder (ComponentBuilder): the component creator that can create components using
                ComponentConfig classes
        """
        self.component_builder = component_builder
        super(AutoUserComponentsSourceSingle, self).__init__(user_type, component_type)

    def get_class(self, name):
        cls = super(AutoUserComponentsSourceSingle, self).get_class(name)
        if issubclass(cls, ComponentConfig):
            return self.component_builder.create_class(cls)
        return cls


class UserComponentsSourceMulti(ComponentsSource):
    """Base class for components in which there are multiple components per file.

    Classes implementing the user components source must overwrite the method: _get_components_from_module()
    used to get the components in a loaded file/module.

    Class Attributes:
        loaded_modules_cache (dict): A cache for loaded components.
            If we do not do this, we fit_model into TypeError problems when a class is reloaded while there is already
            an instantiated object.

            This dict is indexed per directory names and contains for each loaded python file a tuple with the
            module and the loaded component.
    """
    loaded_modules_cache = {}

    def __init__(self, user_type, component_type):
        """
        Args:
            user_type (str): either 'user' or 'standard'. This defines from which dir to use the components
            component_type (str): from which dir in 'user' or 'standard' to use the components
        """
        super(UserComponentsSourceMulti, self).__init__()
        self._user_type = user_type
        self._component_type = component_type

        if self._component_type not in self.loaded_modules_cache:
            self.loaded_modules_cache[self._component_type] = {}

        self.path = _get_components_path(user_type, component_type)
        self._check_path()
        self._components = self._load_all_components()

    def list(self):
        return self._components.keys()

    def get_class(self, name):
        if name not in self._components:
            raise ImportError
        return self._components[name][0]

    def get_meta_info(self, name):
        return self._components[name][1]

    def _load_all_components(self):
        self._update_modules_cache()

        all_components = []
        for module, components in self.loaded_modules_cache[self._component_type].values():
            all_components.extend(components)

        return {meta_info[1]['name']: meta_info for meta_info in all_components}

    def _update_modules_cache(self):
        """Fill the modules cache with the components.

        This loops through all the python files present in the dir name and tries to use them as modules if they
        are not yet loaded.
        """
        for path in self._get_python_component_files():
            if path not in self.loaded_modules_cache[self._component_type]:
                module_name = self._user_type + '/' + \
                              self._component_type + '/' + \
                              os.path.splitext(os.path.basename(path))[0]
                module = imp.load_source(module_name, path)
                self.loaded_modules_cache[self._component_type][path] = [module,
                                                                         self._get_components_from_module(module)]

    def _get_components_from_module(self, module):
        """Return a list of all the available components in the given module.

        Args:
            module (module): the module from which to use the components.

        Returns:
            list: list of components loaded from this module
        """
        loaded_items = []

        if hasattr(module, 'get_components_list'):
            loaded_items.extend(module.get_components_list())

        return loaded_items

    def _get_python_component_files(self):
        return filter(lambda v: os.path.basename(v)[0:2] != '__', glob.glob(os.path.join(self.path, '*.py')))

    def _check_path(self):
        if not os.path.isdir(self.path):
            raise RuntimeError('The components folder "{0}" could not be found. '
                               'Please check the path to the components in your configuration file.'.format(self.path))


class AutoUserComponentsSourceMulti(UserComponentsSourceMulti):

    def __init__(self, user_type, component_type, component_class, component_builder):
        """Create a component source that can create components using multiple types of definitions.

        This will use either objects of the class defined by component_class or it will use objects
        of type ComponentConfig using the builder defined by component_creator or it will use the objects
        from the get_components_list function.

        Args:
            user_type (str): either 'user' or 'standard'. This defines from which dir to use the components
            component_type (str): from which dir in 'user' or 'standard' to use the components
            component_class (class): the class to auto use
            component_builder (ComponentBuilder): the component creator to use for components defined as a
                ComponentConfig.
        """
        self._component_class = component_class
        self.component_builder = component_builder
        super(AutoUserComponentsSourceMulti, self).__init__(user_type, component_type)

    def get_class(self, name):
        if name not in self._components:
            raise ImportError

        base = self._components[name][0]
        if inspect.isclass(base) and issubclass(base, ComponentConfig):
            return self.component_builder.create_class(base)

        return super(AutoUserComponentsSourceMulti, self).get_class(name)

    def _get_components_from_module(self, module):
        """Return a list of all the available components in the given module.

        Args:
            module (module): the module from which to use the components.

        Returns:
            list: list of components loaded from this module
        """
        loaded_items = super(AutoUserComponentsSourceMulti, self)._get_components_from_module(module)

        items = inspect.getmembers(module, _get_class_predicate(module, self._component_class))
        loaded_items.extend((item[1], item[1].meta_info()) for item in items)

        items = inspect.getmembers(module, _get_class_predicate(module, ComponentConfig))
        loaded_items.extend((item[1], item[1].meta_info()) for item in items)

        return loaded_items


class ParametersSource(AutoUserComponentsSourceMulti):

    def __init__(self, user_type):
        """Source for the items in the 'parameters' dir in the components folder."""
        from mot.model_building.cl_functions.parameters import CLFunctionParameter
        from mdt.models.parameters import ParameterBuilder
        super(ParametersSource, self).__init__(user_type, 'parameters', CLFunctionParameter, ParameterBuilder())


class SingleModelSource(AutoUserComponentsSourceMulti):

    def __init__(self, user_type):
        """Source for the items in the 'single_models' dir in the components folder."""
        from mdt.models.single import DMRISingleModel, DMRISingleModelBuilder
        super(SingleModelSource, self).__init__(user_type, 'single_models', DMRISingleModel, DMRISingleModelBuilder())


class CascadeSource(AutoUserComponentsSourceMulti):

    def __init__(self, user_type):
        """Source for the items in the 'cascade_models' dir in the components folder."""
        from mdt.models.cascade import DMRICascadeModelInterface, CascadeBuilder
        super(CascadeSource, self).__init__(user_type, 'cascade_models', DMRICascadeModelInterface, CascadeBuilder())


class MOTSourceSingle(ComponentsSource):

    def get_meta_info(self, name):
        return {}


class MOTLibraryFunctionSource(MOTSourceSingle):

    def get_class(self, name):
        return getattr(mot.model_building.cl_functions.library_functions, name)

    def list(self):
        module = mot.model_building.cl_functions.library_functions
        items = inspect.getmembers(module,  _get_class_predicate(module, LibraryFunction))
        return [x[0] for x in items if x[0] != 'LibraryFunction']


class MOTCompartmentModelsSource(MOTSourceSingle):

    def get_class(self, name):
        return getattr(mot.model_building.cl_functions.model_functions, name)

    def list(self):
        module = mot.model_building.cl_functions.model_functions
        items = inspect.getmembers(module, _get_class_predicate(module, ModelFunction))
        return [x[0] for x in items if x[0] != 'ModelFunction']


class BatchProfilesLoader(ComponentsLoader):

    def __init__(self):
        super(BatchProfilesLoader, self).__init__([UserComponentsSourceSingle('user', 'batch_profiles'),
                                                   UserComponentsSourceSingle('standard', 'batch_profiles')])


class ProcessingStrategiesLoader(ComponentsLoader):

    def __init__(self):
        super(ProcessingStrategiesLoader, self).__init__([UserComponentsSourceSingle('user', 'processing_strategies'),
                                                          UserComponentsSourceSingle('standard', 'processing_strategies')])


class NoiseSTDCalculatorsLoader(ComponentsLoader):

    def __init__(self):
        super(NoiseSTDCalculatorsLoader, self).__init__(
            [UserComponentsSourceSingle('user', 'noise_std_estimators'),
             UserComponentsSourceSingle('standard', 'noise_std_estimators')])


class CompartmentModelsLoader(ComponentsLoader):

    def __init__(self):
        from mdt.models.compartments import CompartmentBuilder
        super(CompartmentModelsLoader, self).__init__(
            [AutoUserComponentsSourceSingle('user', 'compartment_models', CompartmentBuilder()),
             AutoUserComponentsSourceSingle('standard', 'compartment_models', CompartmentBuilder()),
             MOTCompartmentModelsSource()])


class LibraryFunctionsLoader(ComponentsLoader):

    def __init__(self):
        super(LibraryFunctionsLoader, self).__init__([UserComponentsSourceSingle('user', 'library_functions'),
                                                      UserComponentsSourceSingle('standard', 'library_functions'),
                                                      MOTLibraryFunctionSource()])


class SingleModelsLoader(ComponentsLoader):

    def __init__(self):
        super(SingleModelsLoader, self).__init__([SingleModelSource('user'),
                                                  SingleModelSource('standard')])


class ParametersLoader(ComponentsLoader):

    def __init__(self):
        super(ParametersLoader, self).__init__([ParametersSource('user'),
                                                ParametersSource('standard')])


class CascadeModelsLoader(ComponentsLoader):

    def __init__(self):
        super(CascadeModelsLoader, self).__init__([CascadeSource('user'),
                                                   CascadeSource('standard')])


def get_component_class(component_type, component_name):
    """Return the class of the given component.

    Args:
        component_type (str): the type of component, for example 'batch_profiles' or 'parameters'
        component_name (str): the name of the component to use

    Returns:
        the class of the given component
    """
    if component_type == 'batch_profiles':
        return BatchProfilesLoader().get_class(component_name)
    if component_type == 'cascade_models':
        return CascadeModelsLoader().get_class(component_name)
    if component_type == 'compartment_models':
        return CompartmentModelsLoader().get_class(component_name)
    if component_type == 'library_functions':
        return LibraryFunctionsLoader().get_class(component_name)
    if component_type == 'noise_std_estimators':
        return NoiseSTDCalculatorsLoader().get_class(component_name)
    if component_type == 'parameters':
        return ParametersLoader().get_class(component_name)
    if component_type == 'processing_strategies':
        return ProcessingStrategiesLoader().get_class(component_name)
    if component_type == 'single_models':
        return SingleModelsLoader().get_class(component_name)
    raise ValueError('Could not find the given component type {}'.format(component_type))


def load_component(component_type, component_name, *args, **kwargs):
    """Load the class indicated by the given component type and name.

    Args:
        component_type (str): the type of component, for example 'batch_profiles' or 'parameters'
        component_name (str): the name of the component to use
        *args: passed to the component
        **kwargs: passed to the component

    Returns:
        the loaded component
    """
    component = get_component_class(component_type, component_name)
    return component(*args, **kwargs)


def _get_class_predicate(module, class_type):
    """A predicate to be used in the function inspect.getmembers

    This predicate checks if the module of the item we inspect matches the given module, checks the class type
    to be the given class type and checks if the checked item is not in the exclude list.

    Args:
        module (module): the module to check against the module of the item
        class_type (module): the module to check against the class_type of the item

    Returns:
        function: a function to be used as a predicate in inspect.getmembers
    """
    def defined_in_module(item):
        return item.__module__ == module.__name__

    def complete_predicate(item):
        return inspect.isclass(item) and defined_in_module(item) and issubclass(item, class_type)

    return complete_predicate


def _get_components_path(user_type, component_type):
    """
    Args:
        user_type (str): either 'standard' or 'user'
        component_type (str): one of the dir names in standard and user
    """
    from mdt.configuration import get_config_dir
    return os.path.join(get_config_dir(), 'components', user_type, component_type)
