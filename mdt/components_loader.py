import glob
import os
import imp
#todo in P3.4 replace imp calls with importlib.SourceFileLoader(name, path).load_module(name)
from mdt import configuration
from mot.base import LibraryFunction, ModelFunction
import mot.cl_functions
from inspect import getmembers, isclass

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
            sources (list): the list of sources to use
        """
        self._sources = sources

    def list_all(self):
        """List the names of all the available components."""
        return list(sorted(set(self.list_builtin() + self.list_user())))

    def list_builtin(self):
        """List the names of all the loadable builtin components."""
        builtin_modules = []

        for source in self._sources:
            if source.is_builtin:
                builtin_modules.extend(source.list())

        return list(sorted(set(builtin_modules)))

    def list_user(self):
        """List the names of all the components from the user folder."""
        builtin_modules = []

        for source in self._sources:
            if not source.is_builtin:
                builtin_modules.extend(source.list())

        return list(sorted(set(builtin_modules)))

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


class ComponentsLoaderSingle(ComponentsLoader):

    def __init__(self, component_type):
        """The components loader for single component per file component type.

        This expects that the available python files contain a class with the same name as the file in which the class
        resides. For example, if we have a python file named "MySingleComponent.py" we should at least have the class
        named "MySingleComponent" in that python file. Additionally, the file can contain a dictionary named
        'meta_info' which contains meta information about that module.

        To have multiple (named) components in one file you can use the ComponentsLoaderMulti style of
        component loading.

        Args:
            component_type (str): the type of component we wish to load. This should be named exactly to one of the
                directories available in mdt/data/components/
        """
        super(ComponentsLoaderSingle, self).__init__([UserComponentsSourceSingle(component_type),
                                                      MOTSourceSingle(component_type)])


class ComponentsLoaderMulti(ComponentsLoader):

    def __init__(self, component_type):
        """The components loader for multiple components per file component type.

        This expects that the available python files contain a function named "get_components_list()" that should
        return a list of functions that, when called, return a dictionary with information about the available modules.
        That dictionary in turn should at least contain the key 'model_constructor' which should return a function
        that returns a model instance.

        To have a single (named) component in one file you can use the ComponentsLoaderSingle style of
        component loading.

        Args:
            component_type (str): the type of component we wish to load. This should be named exactly to one of the
                directories available in mdt/data/components/
        """
        super(ComponentsLoaderMulti, self).__init__([UserComponentsSourceMulti(component_type)])


class ComponentsSource(object):

    def __init__(self):
        """Defines a source for components.

        This has functions for listing the available components as well as getting the class and meta information.
        """

    def list(self):
        """Get the names of all the available components from this source.
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

    @property
    def is_builtin(self):
        """Check if this source represents builtin modules or loads external modules.

        Returns:
            boolean: if this source is a builtin module, or an external module source
        """
        return False


class ComponentsSourceMulti(ComponentsSource):

    def __init__(self):
        super(ComponentsSourceMulti, self).__init__()
        self._components_list = []

    def list(self):
        return [x['name'] for x in self._components_list]

    def get_class(self, name):
        for x in self._components_list:
            if x['name'] == name:
                return x['model_constructor']
        raise ImportError

    def get_meta_info(self, name):
        for x in self._components_list:
            return {k: v for k, v in x.items() if k != 'model_constructor'}
        return {}


class UserComponentsSourceSingle(ComponentsSource):

    def __init__(self, component_type):
        """
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

    def is_builtin(self):
        return True


class UserComponentsSourceMulti(ComponentsSourceMulti):

    """A cache for loaded components.

    If we do not do this, we fit_model into TypeError problems when the class is reloaded while there is already
    an instantiated object.
    """
    loaded_components = {}

    def __init__(self, component_type):
        super(UserComponentsSourceMulti, self).__init__()
        self._component_type = component_type

        if self._component_type not in self.loaded_components:
            self.loaded_components[self._component_type] = {}

        self.path = os.path.join(os.path.expanduser(configuration.config['components_location']), component_type)
        if not os.path.isdir(self.path):
            raise RuntimeError('The components folder "{0}" could not be found. '
                               'Please check the path to the components in your configuration file.'.format(self.path))

        self._components_list = self._init_components_list()

    def _init_components_list(self):
        if os.path.isdir(self.path):
            components = []

            for path in self._get_python_component_files():
                if path not in self.loaded_components[self._component_type]:
                    module_name = self._component_type + '/' + os.path.basename(path)[0:-3]
                    module = imp.load_source(module_name, path)
                    self.loaded_components[self._component_type][path] = [module, module.get_components_list()]
                components.extend(self.loaded_components[self._component_type][path][1])

            return components
        return []

    def _get_python_component_files(self):
        return filter(lambda v: os.path.basename(v)[0:2] != '__', glob.glob(os.path.join(self.path, '*.py')))

    def is_builtin(self):
        return True

class MOTSourceSingle(ComponentsSource):

    def __init__(self, component_type):
        super(MOTSourceSingle, self).__init__()
        self.component_type = component_type

    def list(self):
        classes = getmembers(mot.cl_functions, isclass)

        if self.component_type == 'library_functions':
            return [x[0] for x in classes if issubclass(x[1], LibraryFunction) and x[0] != 'LibraryFunction']
        elif self.component_type == 'compartments':
            return [x[0] for x in classes if issubclass(x[1], ModelFunction) and x[0] != 'ModelFunction']

        return []

    def get_class(self, name):
        module = mot.cl_functions
        return getattr(module, name)

    def get_meta_info(self, name):
        return {}

    def is_builtin(self):
        return False

class BatchProfilesLoader(ComponentsLoaderSingle):

    def __init__(self):
        super(BatchProfilesLoader, self).__init__('batch_profiles')


class CompartmentModelsLoader(ComponentsLoaderSingle):

    def __init__(self):
        super(CompartmentModelsLoader, self).__init__('compartment_models')


class LibraryFunctionsLoader(ComponentsLoaderSingle):

    def __init__(self):
        super(LibraryFunctionsLoader, self).__init__('library_functions')


class SingleModelsLoader(ComponentsLoaderMulti):

    def __init__(self):
        super(SingleModelsLoader, self).__init__('single_models')


class CascadeModelsLoader(ComponentsLoaderMulti):

    def __init__(self):
        super(CascadeModelsLoader, self).__init__('cascade_models')