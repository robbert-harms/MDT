"""The general components loader.

This modules consists of two main items, component sources and component loaders. The component loaders have a list
of sources from which they load the available components.
"""
import glob
import inspect
import os
import imp
#todo in P3.4 replace imp calls with importlib.SourceFileLoader(name, path).load_module(name)
from mdt import configuration
from mot.base import LibraryFunction, ModelFunction
import mot.cl_functions

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


class ComponentsLoader(object):

    def __init__(self, sources):
        """The base class for loading and displaying components.

        Args:
            sources (list): the list of sources to use for loading the components
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
            name (str): The name of the component we want to load

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
            name (str): The name of the component we want to load

        Returns:
            class or cb function to construct the given component
        """
        source = self._get_preferred_source(name)
        return source.get_class(name)

    def load(self, name, *args, **kwargs):
        """Load the component with the given name

        Args:
            name (str): The name of the component we want to load
            *args: passed to the component
            **args: passed to the component

        Returns:
            the loaded module
        """
        c = self.get_class(name)
        return c(*args, **kwargs)

    def _get_preferred_source(self, name):
        """Try to get the preferred source for the component with the given name.

        The user source takes precedence over the builtin sources.
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
            name (str): The name of the component we want to load

        Returns:
            the construction function
        """
        raise ImportError

    def get_meta_info(self, name):
        """Get the meta information of a component of the given name.

        Args:
            name (str): The name of the component we want to load

        Returns:
            dict: a dictionary with meta information for this component. Standard meta information is:
                - name (str): the name of the component
                - description (str): the description of the component
        """
        return {}


class UserComponentsSourceSingle(ComponentsSource):

    def __init__(self, component_type):
        """

        This expects that the available python files contain a class with the same name as the file in which the class
        resides. For example, if we have a python file named "MySingleComponent.py" we should at least have the class
        named "MySingleComponent" in that python file. Additionally, the file can contain a dictionary named
        'meta_info' which contains meta information about that module.

        Args:
            component_type (str): the type of component we wish to load. This should be named exactly to one of the
                directories available in mdt/data/components/
        """
        super(UserComponentsSourceSingle, self).__init__()
        self.path = os.path.join(os.path.expanduser(configuration.config['components_location']), component_type)

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
                return {}
        return {}


class UserComponentsSourceMulti(ComponentsSource):
    """Base class for components in which there are multiple components per file.

    Inherited classes must overwrite the method: _get_components_from_module() which is used to get the components
    in a loaded file/module.

    Attributes:
        loaded_modules_cache (dict): A cache for loaded components.
            If we do not do this, we fit_model into TypeError problems when a class is reloaded while there is already
            an instantiated object.

            This dict is indexed per directory names and contains for each loaded python file a tuple with the
            module and the loaded component.
    """
    loaded_modules_cache = {}

    def __init__(self, dir_name):
        super(UserComponentsSourceMulti, self).__init__()
        self._dir_name = dir_name

        if self._dir_name not in self.loaded_modules_cache:
            self.loaded_modules_cache[self._dir_name] = {}

        self.path = os.path.join(os.path.expanduser(configuration.config['components_location']), dir_name)
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
        for module, components in self.loaded_modules_cache[self._dir_name].values():
            all_components.extend(components)

        return {meta_info[1]['name']: meta_info for meta_info in all_components}

    def _update_modules_cache(self):
        """Fill the modules cache with the components.

        This loops through all the python files present in the dir name and tries to load them as modules if they
        are not yet loaded.
        """
        for path in self._get_python_component_files():
            if path not in self.loaded_modules_cache[self._dir_name]:
                module_name = self._dir_name + '/' + os.path.splitext(os.path.basename(path))[0]
                module = imp.load_source(module_name, path)
                self.loaded_modules_cache[self._dir_name][path] = [module, self._get_components_from_module(module)]

    def _get_components_from_module(self, module):
        """Return a list of all the available components in the given module.

        Args:
            module (module): the module from which to load the components.

        Returns:
            list: list of components loaded from this module
        """
        items = inspect.getmembers(module, get_class_predicate(module, self._get_desired_class()))
        loaded_items = [(item[1], item[1].meta_info()) for item in items]

        if hasattr(module, 'get_components_list'):
            loaded_items.extend(module.get_components_list())

        return loaded_items

    def _get_desired_class(self):
        """This function is used for the default implementation of _get_components_from_module.

        Returns:
            class: the name of a class we want to look for in the modules.
        """

    def _get_python_component_files(self):
        return filter(lambda v: os.path.basename(v)[0:2] != '__', glob.glob(os.path.join(self.path, '*.py')))

    def _check_path(self):
        if not os.path.isdir(self.path):
            raise RuntimeError('The components folder "{0}" could not be found. '
                               'Please check the path to the components in your configuration file.'.format(self.path))


class SingleModelSource(UserComponentsSourceMulti):

    def __init__(self):
        """Source for the items in the 'single_models' dir in the components folder."""
        super(SingleModelSource, self).__init__('single_models')

    def _get_desired_class(self):
        from mdt.dmri_composite_model import DMRISingleModelBuilder
        return DMRISingleModelBuilder


class CascadeComponentSource(UserComponentsSourceMulti):

    def __init__(self):
        """Source for the items in the 'cascade_models' dir in the components folder."""
        super(CascadeComponentSource, self).__init__('cascade_models')

    def _get_desired_class(self):
        from mdt.cascade_model import CascadeModelInterface
        return CascadeModelInterface


class MOTSourceSingle(ComponentsSource):

    def get_class(self, name):
        module = mot.cl_functions
        return getattr(module, name)

    def get_meta_info(self, name):
        return {}


class MOTLibraryFunctionSource(MOTSourceSingle):

    def list(self):
        module = mot.cl_functions
        items = inspect.getmembers(module, get_class_predicate(module, LibraryFunction))
        return [x[0] for x in items if x[0] != 'LibraryFunction']


class MOTModelsSource(MOTSourceSingle):

    def list(self):
        module = mot.cl_functions
        items = inspect.getmembers(module, get_class_predicate(module, ModelFunction))
        return [x[0] for x in items if x[0] != 'ModelFunction']


class BatchProfilesLoader(ComponentsLoader):

    def __init__(self):
        super(BatchProfilesLoader, self).__init__([UserComponentsSourceSingle('batch_profiles')])


class CompartmentModelsLoader(ComponentsLoader):

    def __init__(self):
        super(CompartmentModelsLoader, self).__init__([UserComponentsSourceSingle('compartment_models'),
                                                       MOTModelsSource()])


class LibraryFunctionsLoader(ComponentsLoader):

    def __init__(self):
        super(LibraryFunctionsLoader, self).__init__([UserComponentsSourceSingle('library_functions'),
                                                      MOTLibraryFunctionSource()])


class SingleModelsLoader(ComponentsLoader):

    def __init__(self):
        super(SingleModelsLoader, self).__init__([SingleModelSource()])


class CascadeModelsLoader(ComponentsLoader):

    def __init__(self):
        super(CascadeModelsLoader, self).__init__([CascadeComponentSource()])


def get_class_predicate(module, class_type):
    """A predicate to be used in the function inspect.getmembers

    This predicate checks if the module of the item we inspect matches the given module, checks the class type
    to be the given class type and checks if the checked item is not in the exclude list.

    Args:
        module (module): the module to check against the module of the item
        class_type (module): the module to check against the class_type of the item

    Returns:
        function: a function to be used as a predicate in inspect.getmembers
    """
    defined_in_module = lambda item: item.__module__ == module.__name__
    is_subclass = lambda item: issubclass(item, class_type)

    def predicate_function(item):
        return inspect.isclass(item) and defined_in_module(item) and is_subclass(item)

    return predicate_function
