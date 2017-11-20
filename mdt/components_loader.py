"""The general components loader.

This modules consists of two main items, component sources and component loaders. The component loaders have a list
of sources from which they load the available components.
"""
import collections
import imp  # todo in P3.4 replace imp calls with importlib.SourceFileLoader(name, path).load_module(name)
import inspect
import os
from contextlib import contextmanager

import mot.library_functions
import mot.model_building.model_functions
from mdt.component_templates.base import ComponentTemplate, construct_component
from mdt.exceptions import NonUniqueComponent
from mot.library_functions import SimpleCLLibrary
from mot.model_building.likelihood_functions import LikelihoodFunction
from mot.model_building.model_functions import ModelCLFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_model(model_name, **kwargs):
    """Load one of the available models.

    Args:
        model_name (str): One of the models from get_list_of_composite_models() or get_list_of_cascade_models()
        **kwargs: Extra keyword arguments used for the initialization of the model

    Returns:
        Either a cascade model or a composite model. In any case, a model that can be given to the fit_model function.
    """
    cml = CascadeModelsLoader()
    sml = CompositeModelsLoader()
    try:
        return cml.load(model_name, **kwargs)
    except ImportError:
        try:
            return sml.load(model_name, **kwargs)
        except ImportError:
            raise ValueError('The model with the name "{}" could not be found.'.format(model_name))


def get_compartment(compartment_name, **kwargs):
    """Load one of the available compartments.

    Args:
        compartment_name (str): the name of the compartment to load
        **kwargs: Extra keyword arguments used for the initialization of the model

    Returns:
        mdt.models.compartments.DMRICompartmentModelFunction: the compartment model requested
    """
    return CompartmentModelsLoader().load(compartment_name, **kwargs)


def get_parameter(parameter_name, **kwargs):
    """Load one of the available compartments.

    Args:
        parameter_name (str): the name of the parameter to load
        **kwargs: Extra keyword arguments used for the initialization of the model

    Returns:
        the compartment model requested
    """
    return ParametersLoader().load(parameter_name, **kwargs)


def get_library_function(function_name, **kwargs):
    """Load one of the available compartments.

    Args:
        function_name (str): the name of the library function to load
        **kwargs: Extra keyword arguments used for the initialization of the model

    Returns:
        the library function requested
    """
    return LibraryFunctionsLoader().load(function_name, **kwargs)


@contextmanager
def user_preferred_components(components_dict):
    """Creates a context manager in which the provided components take precedence over existing ones.

    Useful if you want to monkey patch certain parts of a model.

    Args:
        components_dict (dict): dictionary with the structure ``{<component_type>: {<name>: <cls>, ...}, ...}``
    """
    UserPreferredSource.add_components(components_dict)
    yield
    UserPreferredSource.remove_components(components_dict)


class ComponentsLoader(object):

    def __init__(self, sources):
        """The base class for loading and displaying components.

        Args:
            sources (list of ComponentsSource): the list of sources to use for loading the components
        """
        self._sources = sources
        self._check_unique_names()

    def list_all(self):
        """List the names of all the available components.

        Returns:
            list of str: the list of available components by name
        """
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

    def reload(self):
        """Tell this component loader to empty the cache and reload the modules."""
        for source in self._sources:
            source.reload()

    def _get_preferred_source(self, name):
        """Try to get the preferred source for the component with the given name.

        The order of the sources matter, the first source takes precedence over the latter ones and so forth.

        Returns:
            ComponentsSource: the source from where to load the given component
        """
        for source in self._sources:
            try:
                source.get_class(name)
                return source
            except ImportError:
                pass
        raise ImportError("No component found with the name {}".format(name))

    def _check_unique_names(self):
        """Check if all the elements in the sources are unique.

        Raises:
            NonUniqueComponent: if two components with the same name are found.
        """
        elements = []
        for source in self._sources:
            if source.use_in_uniqueness_check():
                elements.extend(source.list())
        non_unique = list(item for item, count in collections.Counter(elements).items() if count > 1)
        if len(non_unique):
            raise NonUniqueComponent('Non-unique components detected: {}, please rename them.'.format(non_unique))


class ComponentsSource(object):
    """Defines a source for components.

    This has functions for listing the available components as well as getting the class and meta information.
    """

    def list(self):
        """Get the names of all the available components from this source.

        Returns:
            list or str: list of the names of all the components loadable from this source.
        """
        raise NotImplementedError()

    def get_class(self, name):
        """Get the class for the component by the given name

        Args:
            name (str): The name of the component we want to use

        Returns:
            the construction function
        """
        raise NotImplementedError()

    def get_meta_info(self, name):
        """Get the meta information of a component of the given name.

        Args:
            name (str): The name of the component we want to use

        Returns:
            dict: a dictionary with meta information for this component. Standard meta information is:
                - name (str): the name of the component
                - description (str): the description of the component
        """
        raise NotImplementedError()

    def use_in_uniqueness_check(self):
        """If this source should be used when checking for unique components over all sources.

        By default this is set to True to be able to warn the user if multiple components are found. There are
        cases however in which you may want to disable this feature for a component source such that you can overwrite
        existing components.

        Returns:
            bool: if this class should be used when checking for unique components
        """
        raise NotImplementedError()

    def reload(self):
        """Tell this source to empty the cache and reload the modules."""
        raise NotImplementedError()


class UserPreferredSource(ComponentsSource):

    _components = {}

    def __init__(self, component_type):
        """A source for user defined preferred components.

        To function correctly, this source should be set as the primary source for a component loader. Then,
        the user can use the static methods of this class to add component items on the fly. This allows for example
        'monkey patching' some components by replacing it with a preferred component prior to loading.

        To use, add a new instance of this source to one of your components loaders while ensuring that the
        ``component_type`` is set correctly. Then, using class attributes the correct class elements are returned when
        requested.

        Args:
            component_type (str): the component type for this source
        """
        super(UserPreferredSource, self).__init__()
        self._component_type = component_type

    def list(self):
        return list(self._components.get(self._component_type, {}))

    def get_class(self, name):
        components = self._components.get(self._component_type, {})
        if name in components:
            return components[name]
        raise ImportError

    def use_in_uniqueness_check(self):
        return False

    def get_meta_info(self, name):
        return {}

    def reload(self):
        pass

    @classmethod
    def add_component(cls, component_type, name, component_class):
        """Add a component to the general user preferred component source.

        This class is designed to work with components of various types where the instance of an class determines which
        component is loaded. Hence, when adding a component we need to set the component type.

        Args:
            component_type (str): the type of this component
            name (str): the name of this component
            component_class (cls): the class to load when this component / name is required.
        """
        if component_type not in cls._components:
            cls._components[component_type] = {}
        cls._components[component_type][name] = component_class

    @classmethod
    def add_components(cls, components_dict):
        """Add one or more components to this source using a dictionary.

        This will load all provided components in the given dictionary.

        Args:
            components_dict (dict): dictionary with the structure ``{<component_type>: {<name>: <cls>, ...}, ...}``
        """
        for component_type in components_dict:
            for name, component_class in components_dict[component_type].items():
                cls.add_component(component_type, name, component_class)

    @classmethod
    def remove_component(cls, component_type, name):
        """Remove a component from the general user preferred component source.

        Args:
            component_type (str): the type of this component
            name (str): the name of this component
        """
        if component_type in cls._components:
            if name in cls._components[component_type]:
                del cls._components[component_type][name]

    @classmethod
    def remove_components(cls, components_dict):
        """Remove one or more components from this source using a dictionary.

        Args:
            components_dict (dict): dictionary with the structure ``{<component_type>: [<name>, ...], ...}``
                That is, per component type a list of components to remove.
        """
        for component_type in components_dict:
            for name in components_dict[component_type]:
                cls.remove_component(component_type, name)


class AutomaticCascadeSource(ComponentsSource):

    _can_create_list = True  # to prevent double recursion

    def __init__(self):
        """A source that automatically creates cascade models if there is no existing cascade.

        This generates cascade models matching the scheme in Harms 2017: CS, CI and CF cascades.

        The first type of cascades generates are the Cascade S0 schemes. In this we initialize the desired model
        using an estimate of the unweighted signal S0 (sometimes called B0). Creating cascades for this type is easy
        since it is a two component cascade.
        """
        super(AutomaticCascadeSource, self).__init__()

    def list(self):
        from mdt.configuration import use_automatic_generated_cascades, get_automatic_generated_cascades_excluded
        if use_automatic_generated_cascades():
            if AutomaticCascadeSource._can_create_list:
                AutomaticCascadeSource._can_create_list = False
                models_list = CompositeModelsLoader().list_all()
                cascades_list = CascadeModelsLoader().list_all()
                AutomaticCascadeSource._can_create_list = True

                excludes = get_automatic_generated_cascades_excluded()
                models_list = [m for m in models_list if m not in excludes]

                return_list = []
                return_list.extend(self._get_missing_s0_cascades(models_list, cascades_list))

                return return_list
        return []

    def get_class(self, name):
        available_models = self.list()
        if name in available_models:
            return construct_component(self._generate_cascade(name))

        raise ImportError

    def use_in_uniqueness_check(self):
        return True

    def get_meta_info(self, name):
        template = self._generate_cascade(name)
        return template.meta_info()

    def reload(self):
        pass

    def _get_missing_s0_cascades(self, models, cascades):
        """Get the list of cascade model names that are missing from the list of cascades.

        Args:
            models (list): the list of composite model names
            cascades (list): the list of existing cascaded model names

        Returns:
            list: the list with cascade models that may be missing
        """
        missing_cascades = []
        for model_name in models:
            if '{} (Cascade|S0)'.format(model_name) not in cascades:
                missing_cascades.append('{} (Cascade|S0)'.format(model_name))
        return missing_cascades

    def _generate_cascade(self, cascaded_name):
        """Generate the cascade for the given name.

        The provided name has the format "<model> (Cascade[|S0]?)". This class will create the right model based
        on the cascade extension.

        Args:
            cascaded_name (str): the model name we are going to generate
        """
        from mdt.component_templates.cascade_models import CascadeTemplate

        if '(Cascade|S0)' in cascaded_name:
            class template(CascadeTemplate):
                cascade_name_modifier = 'S0'
                description = 'Automatically generated cascade.'
                models = ('S0',
                          cascaded_name[0:-len('(Cascade|S0)')].strip())
            return template

        raise ImportError


class UserComponentsSourceSingle(ComponentsSource):

    loaded_modules_cache = {}

    def __init__(self, user_type, component_type):
        """Load the user components of the type *single*.

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
        self._class_filenames = {}

    def list(self):
        items = []
        if os.path.isdir(self.path):
            for dir_name, sub_dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        items.append(file[0:-3])
                        self._class_filenames[file[0:-3]] = os.path.join(dir_name, file)
        return items

    def get_class(self, name):
        if name in self._class_filenames:
            if name not in self.loaded_modules_cache:
                module = imp.load_source(name, self._class_filenames[name])
                self.loaded_modules_cache[name] = (module, getattr(module, name))
            return self.loaded_modules_cache[name][1]
        raise ImportError

    def get_meta_info(self, name):
        if name in self._class_filenames:
            if name not in self.loaded_modules_cache:
                module = imp.load_source(name, self._class_filenames[name])
                self.loaded_modules_cache[name] = (module, getattr(module, name))
            try:
                return getattr(self.loaded_modules_cache[name][-1], 'meta_info')()
            except AttributeError:
                try:
                    cls = self.get_class(name)
                    return cls.meta_info()
                except AttributeError:
                    return {}
        return {}

    def reload(self):
        self.__class__.loaded_modules_cache = {}
        self._class_filenames = {}

    def use_in_uniqueness_check(self):
        return True


class AutoUserComponentsSourceSingle(UserComponentsSourceSingle):

    def __init__(self, user_type, component_type, component_builder):
        """

        This class extends the default single components source loader by also being able to use components defined
        using the ComponentTemplate method. This means that the components are defined as subclasses of ComponentTemplate
        and we need a ComponentBuilder to actually create the components.

        Args:
            user_type (str): either 'standard' or 'user'
            component_type (str): the type of component we wish to use. This should be named exactly to one of the
                directories available in mdt/data/components/
            component_builder (mdt.component_templates.base.ComponentBuilder): the component creator that can create components using
                ComponentTemplate classes
        """
        self.component_builder = component_builder
        super(AutoUserComponentsSourceSingle, self).__init__(user_type, component_type)

    def get_class(self, name):
        cls = super(AutoUserComponentsSourceSingle, self).get_class(name)
        if issubclass(cls, ComponentTemplate):
            return self.component_builder.create_class(cls)
        return cls


class ComponentInfo(object):

    def get_name(self):
        """Get the name of this component

        Returns:
            str: the name of this component
        """
        return NotImplemented

    def get_meta_info(self):
        """Get the additional meta info of this component

        Returns:
            dict: the meta info
        """
        return NotImplemented

    def get_component_class(self):
        """Get the component class

        Returns:
            class: the class of the component
        """
        return NotImplemented


class UserComponentsSourceMulti(ComponentsSource):

    loaded_modules_cache = {}

    def __init__(self, user_type, component_type):
        """"Base class for components in which there are multiple components per file.

        Args:
            user_type (str): either 'user' or 'standard'. This defines from which dir to use the components
            component_type (str): from which dir in 'user' or 'standard' to use the components

        Attributes:
            loaded_modules_cache (dict): A cache for loaded components.
                If we do not do this, we fit_model into TypeError problems when a class is reloaded while there is
                already an instantiated object.

                This dict is indexed per directory names and contains for each loaded python file a tuple with the
                module and the loaded component.
        """
        super(UserComponentsSourceMulti, self).__init__()
        self._user_type = user_type
        self._component_type = component_type
        self._path = _get_components_path(user_type, component_type)
        self._check_path()
        self._components = self._load_all_components()

    def list(self):
        return self._components.keys()

    def get_class(self, name):
        if name not in self._components:
            raise ImportError
        return self._components[name].get_component_class()

    def get_meta_info(self, name):
        return self._components[name].get_meta_info()

    def reload(self):
        self.__class__.loaded_modules_cache = {}
        self._load_all_components()

    def use_in_uniqueness_check(self):
        return True

    def _load_all_components(self):
        self._update_modules_cache()

        all_components = []
        for module, components in self.loaded_modules_cache[self._user_type][self._component_type].values():
            all_components.extend(components)

        return {component.get_name(): component for component in all_components}

    def _update_modules_cache(self):
        """Fill the modules cache with the components.

        This loops through all the python files present in the dir name and tries to use them as modules if they
        are not yet loaded.
        """
        if self._user_type not in self.loaded_modules_cache:
            self.loaded_modules_cache[self._user_type] = {}

        if self._component_type not in self.loaded_modules_cache[self._user_type]:
            self.loaded_modules_cache[self._user_type][self._component_type] = {}

        for dir_name, sub_dirs, files in os.walk(self._path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    path = os.path.join(dir_name, file)

                    if path not in self.loaded_modules_cache[self._user_type][self._component_type]:
                        module_name = self._user_type + '/' + \
                                      self._component_type + '/' + \
                                      dir_name[len(self._path) + 1:] + '/' + \
                                      os.path.splitext(os.path.basename(path))[0]

                        module = imp.load_source(module_name, path)

                        self.loaded_modules_cache[self._user_type][self._component_type][path] = \
                            (module, self._get_components_from_module(module))

    def _get_components_from_module(self, module):
        """Return a list of all the available components in the given module.

        Args:
            module (module): the module from which to use the components.

        Returns:
            list: list of ComponentInfo objects
        """
        loaded_items = []

        if hasattr(module, 'get_components_list'):
            loaded_items.extend(module.get_components_list())

        return loaded_items

    def _check_path(self):
        if not os.path.isdir(self._path):
            raise RuntimeError('The components folder "{0}" could not be found. '
                               'Please check the path to the components in your configuration file.'.format(self._path))


class AutoUserComponentsSourceMulti(UserComponentsSourceMulti):

    def __init__(self, user_type, component_type, component_class, component_builder):
        """Create a component source that can create components using multiple types of definitions.

        This will use either objects of the class defined by component_class or it will use objects
        of type ComponentTemplate using the builder defined by component_creator or it will use the objects
        from the get_components_list function.

        Args:
            user_type (str): either 'user' or 'standard'. This defines from which dir to use the components
            component_type (str): from which dir in 'user' or 'standard' to use the components
            component_class (class): the class to auto use
            component_builder (mdt.component_templates.base.ComponentBuilder): the component creator to use for
                components defined as a ComponentTemplate.
        """
        self._component_class = component_class
        self.component_builder = component_builder
        super(AutoUserComponentsSourceMulti, self).__init__(user_type, component_type)

    def get_class(self, name):
        if name not in self._components:
            raise ImportError

        base = self._components[name].get_component_class()
        if inspect.isclass(base) and issubclass(base, ComponentTemplate):
            return self.component_builder.create_class(base)

        return super(AutoUserComponentsSourceMulti, self).get_class(name)

    def _get_components_from_module(self, module):
        """Return a list of all the available components in the given module.

        Args:
            module (module): the module from which to use the components.

        Returns:
            list: list of components loaded from this module
        """
        class DynamicInfo(ComponentInfo):

            def __init__(self, component):
                self._component = component

            def get_name(self):
                return self._component.name

            def get_meta_info(self):
                return self._component.meta_info()

            def get_component_class(self):
                return self._component

        loaded_items = super(AutoUserComponentsSourceMulti, self)._get_components_from_module(module)

        items = inspect.getmembers(module, _get_class_predicate(module, self._component_class))
        loaded_items.extend(DynamicInfo(item[1]) for item in items)

        items = inspect.getmembers(module, _get_class_predicate(module, ComponentTemplate))
        loaded_items.extend(DynamicInfo(item[1]) for item in items)

        return loaded_items


class ParametersSource(AutoUserComponentsSourceMulti):

    def __init__(self, user_type):
        """Source for the items in the 'parameters' dir in the components folder."""
        from mot.cl_parameter import CLFunctionParameter
        from mdt.component_templates.parameters import ParameterBuilder
        super(ParametersSource, self).__init__(user_type, 'parameters', CLFunctionParameter, ParameterBuilder())


class CompositeModelSource(AutoUserComponentsSourceMulti):

    def __init__(self, user_type):
        """Source for the items in the 'composite_models' dir in the components folder."""
        from mdt.models.composite import DMRICompositeModel
        from mdt.component_templates.composite_models import DMRICompositeModelBuilder
        super(CompositeModelSource, self).__init__(user_type, 'composite_models', DMRICompositeModel,
                                                   DMRICompositeModelBuilder())


class CompartmentModelSource(AutoUserComponentsSourceMulti):

    def __init__(self, user_type):
        """Source for the items in the 'composite_models' dir in the components folder."""
        from mdt.models.compartments import DMRICompartmentModelFunction
        from mdt.component_templates.compartment_models import CompartmentBuilder
        super(CompartmentModelSource, self).__init__(user_type, 'compartment_models', DMRICompartmentModelFunction,
                                                     CompartmentBuilder())


class LibraryFunctionSource(AutoUserComponentsSourceMulti):

    def __init__(self, user_type):
        """Source for the items in the 'composite_models' dir in the components folder."""
        from mot.library_functions import CLLibrary
        from mdt.component_templates.library_functions import LibraryFunctionsBuilder
        super(LibraryFunctionSource, self).__init__(user_type, 'library_functions', CLLibrary,
                                                    LibraryFunctionsBuilder())


class CascadeSource(AutoUserComponentsSourceMulti):

    def __init__(self, user_type):
        """Source for the items in the 'cascade_models' dir in the components folder."""
        from mdt.models.cascade import DMRICascadeModelInterface
        from mdt.component_templates.cascade_models import CascadeBuilder
        super(CascadeSource, self).__init__(user_type, 'cascade_models', DMRICascadeModelInterface, CascadeBuilder())


class MOTSourceSingle(ComponentsSource):

    def list(self):
        raise NotImplementedError()

    def get_class(self, name):
        raise NotImplementedError()

    def get_meta_info(self, name):
        return {}

    def reload(self):
        pass

    def use_in_uniqueness_check(self):
        return True


class MOTLibraryFunctionSource(MOTSourceSingle):

    def get_class(self, name):
        try:
            return getattr(mot.library_functions, name)
        except AttributeError:
            raise ImportError

    def list(self):
        module = mot.library_functions
        items = inspect.getmembers(module, _get_class_predicate(module, SimpleCLLibrary))
        return [x[0] for x in items if x[0] != 'SimpleCLLibrary']


class MOTCompartmentModelsSource(MOTSourceSingle):

    def get_class(self, name):
        try:
            return getattr(mot.model_building.model_functions, name)
        except AttributeError:
            raise ImportError

    def list(self):
        module = mot.model_building.model_functions
        items = inspect.getmembers(module, _get_class_predicate(module, ModelCLFunction))
        return [x[0] for x in items if x[0] != 'ModelCLFunction']


class MOTLikelihoodFunctionSource(MOTSourceSingle):

    def get_class(self, name):
        try:
            return getattr(mot.model_building.likelihood_functions, name)
        except AttributeError:
            raise ImportError

    def list(self):
        module = mot.model_building.likelihood_functions
        items = inspect.getmembers(module, _get_class_predicate(module, LikelihoodFunction))
        return [x[0] for x in items if x[0] != 'LikelihoodFunction']


class BatchProfilesLoader(ComponentsLoader):

    def __init__(self):
        super(BatchProfilesLoader, self).__init__(
            [UserPreferredSource('batch_profiles'),
             UserComponentsSourceSingle('standard', 'batch_profiles'),
             UserComponentsSourceSingle('user', 'batch_profiles')])


class NoiseSTDCalculatorsLoader(ComponentsLoader):

    def __init__(self):
        super(NoiseSTDCalculatorsLoader, self).__init__(
            [UserPreferredSource('noise_std_estimators'),
             UserComponentsSourceSingle('standard', 'noise_std_estimators'),
             UserComponentsSourceSingle('user', 'noise_std_estimators')])


class CompartmentModelsLoader(ComponentsLoader):

    def __init__(self):
        super(CompartmentModelsLoader, self).__init__(
            [UserPreferredSource('compartment_models'),
             CompartmentModelSource('standard'),
             CompartmentModelSource('user'),
             MOTCompartmentModelsSource()])


class LibraryFunctionsLoader(ComponentsLoader):

    def __init__(self):
        super(LibraryFunctionsLoader, self).__init__(
            [UserPreferredSource('library_functions'),
             LibraryFunctionSource('standard'),
             LibraryFunctionSource('user'),
             MOTLibraryFunctionSource()])


class LikelihoodFunctionsLoader(ComponentsLoader):

    def __init__(self):
        super(LikelihoodFunctionsLoader, self).__init__(
            [MOTLikelihoodFunctionSource()])


class CompositeModelsLoader(ComponentsLoader):

    def __init__(self):
        super(CompositeModelsLoader, self).__init__(
            [UserPreferredSource('composite_models'),
             CompositeModelSource('standard'),
             CompositeModelSource('user')])


class ParametersLoader(ComponentsLoader):

    def __init__(self):
        super(ParametersLoader, self).__init__(
            [UserPreferredSource('parameters'),
             ParametersSource('standard'),
             ParametersSource('user')])


class CascadeModelsLoader(ComponentsLoader):

    def __init__(self):
        super(CascadeModelsLoader, self).__init__(
            [UserPreferredSource('cascade_models'),
             CascadeSource('standard'),
             CascadeSource('user'),
             AutomaticCascadeSource()])


def get_meta_info(component_type, component_name):
    """Get the meta info of the requested component.

    Args:
        component_type (str): the type of component, for example 'batch_profiles' or 'parameters'
        component_name (str): the name of the component to use

    Returns:
        dict: the meta information of the given component
    """
    return get_loader(component_type).get_meta_info(component_name)


def get_component_class(component_type, component_name):
    """Return the class of the requested component.

    Args:
        component_type (str): the type of component, for example 'batch_profiles' or 'parameters'
        component_name (str): the name of the component to use

    Returns:
        class: the class of the given component
    """
    return get_loader(component_type).get_class(component_name)


def get_loader(component_type):
    """Get the loader corresponding to the given component type.

    Args:
        component_type (str): the type of component we want the loader for

    Returns:
        ComponentsLoader: the loader corresponding to the component type
    """
    if component_type == 'batch_profiles':
        return BatchProfilesLoader()
    if component_type == 'cascade_models':
        return CascadeModelsLoader()
    if component_type == 'composite_models':
        return CompositeModelsLoader()
    if component_type == 'compartment_models':
        return CompartmentModelsLoader()
    if component_type == 'library_functions':
        return LibraryFunctionsLoader()
    if component_type == 'noise_std_estimators':
        return NoiseSTDCalculatorsLoader()
    if component_type == 'parameters':
        return ParametersLoader()
    if component_type == 'likelihood_functions':
        return LikelihoodFunctionsLoader()
    raise ValueError('Could not find a loader for the given component type "{}"'.format(component_type))


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


def component_import(import_str, component_name):
    """Load one of the components as a module and return the desired component from that module.

    This is in usage similar to 'from ... import ...' except that we use a function in this case.
    You can use it to import raw template classes from the components folders to use as super classes. Example::

        from mdt import component_import

        class MyCHARMED_r1(component_import('standard.composite_models.CHARMED', 'CHARMED_r1')):
            pass

    Here we create a new CHARMED_r1 template using the standard CHARMED_r1 template as a basis.

    Args:
        import_str (str): the import string, something like 'standard.composite_models.CHARMED'.
        component_name (str): the component to load from the virtually imported module

    Returns:
        object: the loaded object from the given module
    """
    def get_component_path(import_str):
        import_items = import_str.split('.')
        component_path = _get_components_path(import_items[0], import_items[1])

        for ind, import_item in enumerate(import_items[2:-1]):
            component_path = os.path.join(component_path, import_item)

        return os.path.join(component_path, import_items[-1] + '.py')

    module_name = 'mdt.virtual_components.' + import_str
    module = imp.load_source(module_name, get_component_path(import_str))

    items = inspect.getmembers(module)

    for name, item in items:
        if name == component_name:
            return item


def _get_class_predicate(desired_module, class_type):
    """A predicate to be used in the function inspect.getmembers

    This predicate checks if the module of the item we inspect matches the given module and checks if the class type
    is of the given class type.

    Args:
        desired_module (module): the module to check against the module of the item
        class_type (Type): the module to check against the class_type of the item

    Returns:
        function: a function to be used as a predicate in inspect.getmembers
    """
    def defined_in_module(item):
        return item.__module__ == desired_module.__name__

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
