import logging
from importlib.machinery import SourceFileLoader
import inspect
import os
from collections import defaultdict
from contextlib import contextmanager
import mdt
import mot
from mdt.configuration import get_config_dir
from mot.library_functions import CLLibrary
from mdt.model_building.likelihood_functions import LikelihoodFunction
from mdt.model_building.parameter_functions.transformations import AbstractTransformation

__author__ = 'Robbert Harms'
__date__ = '2018-03-22'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


supported_component_types = ('batch_profiles', 'cascade_models', 'compartment_models',
                             'composite_models', 'library_functions', 'parameters', 'likelihood_functions',
                             'parameter_transforms')


class _ComponentLibrary:

    def __init__(self):
        """Holds the reference to all defined components, by component type and by name.

        For each component type several components may be defined with different or equal names. If the names are equal
        they are added to a stack and only the last element is returned. Components may also be removed again from
        the stack (in a random access method).
        """
        self._library = {}
        self._mutation_history = []
        self.reset()

    def get_current_history_length(self):
        """Get the current length of the history. Useful for undoing history changes."""
        return len(self._mutation_history)

    def undo_history_until(self, history_ind):
        """Pop the history stack until we are at the given index (length).

        Args:
            history_ind (int): the desired length of the history stack
        """
        if history_ind < len(self._mutation_history):
            for ind in range(len(self._mutation_history) - history_ind):
                item = self._mutation_history.pop()
                if item.action == 'add':
                    self._library[item.component_type][item.name].pop()
                if item.action == 'remove':
                    self._library[item.component_type][item.name].append(item.adapter)

    def reset(self):
        """Clear the library by removing all available components.
        This also resets the mutation history.
        """
        self._library = {component_type: defaultdict(list) for component_type in supported_component_types}
        self._mutation_history = []

    def add_component(self, component_type, name, component_class, meta_info=None):
        """Adds a component class to the library.

        Args:
            component_type (str): the type of the component, see ``supported_component_types``.
            name (str): the name of the component
            component_class (class): the class or constructor function for the component
            meta_info (dict): a dictionary with meta information about the component
        """
        adapter = _DirectComponent(component_class, meta_info=meta_info)
        self._library[component_type][name].append(adapter)
        self._mutation_history.append(_LibraryHistoryDelta('add', component_type, name, adapter))

    def add_template_component(self, template):
        """Adds a component template to the library.

        Args:
            template (mdt.component_templates.base.ComponentTemplateMeta): the template for constructing the component
                class.
        """
        adapter = _ComponentFromTemplate(template)
        self._library[template.component_type][template.name].append(adapter)
        self._mutation_history.append(_LibraryHistoryDelta('add', template.component_type, template.name, adapter))

    def get_component(self, component_type, name):
        """Get the component class for the component of the given type and name.

        Args:
            component_type (str): the type of the component, see ``supported_component_types``.
            name (str): the name of the component

        Returns:
            class: the component class.
        """
        if not self.has_component(component_type, name):
            raise ValueError('Can not find a component of type "{}" with name "{}"'.format(component_type, name))
        return self._library[component_type][name][-1].get_component()

    def get_meta_info(self, component_type, name):
        """Get the meta information dictionary for the component of the given type and name.

        Args:
            component_type (str): the type of the component, see ``supported_component_types``.
            name (str): the name of the component

        Returns:
            dict: the meta information
        """
        if not self.has_component(component_type, name):
            raise ValueError('Can not find a component of type "{}" with name "{}"'.format(component_type, name))
        return self._library[component_type][name][-1].get_meta_info()

    def get_component_list(self, component_type):
        """Get a list of available components by component type.

        Args:
            component_type (str): the type of the component, see ``supported_component_types``.

        Returns:
            list of str: list of available components
        """
        return list(self._library[component_type].keys())

    def has_component(self, component_type, name):
        """Check if a component is available.

        Args:
            component_type (str): the type of the component, see ``supported_component_types``.
            name (str): the name of the component

        Returns:
            boolean: if we have a component available of the given type and given name.
        """
        return name in self._library[component_type] and len(self._library[component_type][name])

    def get_template(self, component_type, name):
        """Get the template class for the given component.

        This may not be supported for all component types and components. That is, since components can either be
        added as classes or as templates, we can not guarantee a template class for any requested component.

        Args:
            component_type (str): the type of the component, see ``supported_component_types``.
            name (str): the name of the component

        Returns:
            mdt.component_templates.base.ComponentTemplateMeta: a template class if possible.

        Raises:
            ValueError: if no component of the given name could be found.
        """
        if not self.has_component(component_type, name):
            raise ValueError('The component with the name "{}" '
                             'of type "{}" could be found.'.format(name, component_type))
        return self._library[component_type][name][-1].get_template()

    def remove_last_entry(self, component_type, name):
        """Removes the last entry of the given component.

        Args:
            component_type (str): the type of the component, see ``supported_component_types``.
            name (str): the name of the component
        """
        adapter = self._library[component_type][name].pop()
        if not len(self._library[component_type][name]):
            del self._library[component_type][name]
        self._mutation_history.append(_LibraryHistoryDelta('remove', component_type, name, adapter))


class _ComponentAdapter:

    def get_component(self):
        """Build or return the actual component class.

        Since the component library supports both ``component classes`` as ``template classes`` we need an adapter class
        to build the actual component if necessary.

        Returns:
            class: the component class
        """
        raise NotImplementedError()

    def get_meta_info(self):
        """Get the meta info of this component

        Returns:
            dict: the meta info
        """
        raise NotImplementedError()

    def get_template(self):
        """If supported, gets the template of this component.

        Returns:
            mdt.component_templates.base.ComponentTemplateMeta: a template class if possible.
        """
        raise NotImplementedError()


class _DirectComponent(_ComponentAdapter):

    def __init__(self, component, meta_info=None):
        self.component = component
        self.meta_info = meta_info or {}

    def get_component(self):
        return self.component

    def get_meta_info(self):
        return self.meta_info

    def get_template(self):
        raise ValueError('Can not build template from component class.')


class _ComponentFromTemplate(_ComponentAdapter):

    def __init__(self, template):
        self.template = template

    def get_component(self):
        return self.template()

    def get_meta_info(self):
        return self.template.meta_info()

    def get_template(self):
        return self.template


class _LibraryHistoryDelta:

    def __init__(self, action, component_type, name, adapter):
        """Representation of a history change in the component library.

        Args:
            action (str): one of ``remove`` or ``add``.
            component_type (str): the type of the component
            name (str): the name of the component.
            adapter (_ComponentAdapter): the adapter instance
        """
        self.component_type = component_type
        self.name = name
        self.adapter = adapter
        self.action = action


component_library = _ComponentLibrary()


def _add_doc(value):
    """Add a docstring to the given value."""
    def _doc(func):
        func.__doc__ = value
        return func
    return _doc


@_add_doc(_ComponentLibrary.add_component.__doc__)
def add_component(component_type, name, cls, meta_info=None):
    return component_library.add_component(component_type, name, cls, meta_info)


@_add_doc(_ComponentLibrary.add_template_component.__doc__)
def add_template_component(template):
    return component_library.add_template_component(template)


@_add_doc(_ComponentLibrary.get_template.__doc__)
def get_template(component_type, name):
    return component_library.get_template(component_type, name)


@_add_doc(_ComponentLibrary.get_component.__doc__)
def get_component(component_type, name):
    return component_library.get_component(component_type, name)


@_add_doc(_ComponentLibrary.has_component.__doc__)
def has_component(component_type, name):
    return component_library.has_component(component_type, name)


@_add_doc(_ComponentLibrary.get_component_list.__doc__)
def get_component_list(component_type):
    return component_library.get_component_list(component_type)


@_add_doc(_ComponentLibrary.get_meta_info.__doc__)
def get_meta_info(component_type, name):
    return component_library.get_meta_info(component_type, name)


@_add_doc(_ComponentLibrary.remove_last_entry.__doc__)
def remove_last_entry(component_type, name):
    return component_library.remove_last_entry(component_type, name)


@contextmanager
def temporary_component_updates():
    """Creates a context that keeps track of the component mutations and undoes them when the context exits.

    This can be useful to temporarily add or remove some components from the library.
    """
    history_ind = component_library.get_current_history_length()
    yield
    component_library.undo_history_until(history_ind)


def reload():
    """Clear the component library and reload all default components.

    This will load the components from the user home folder and from the MOT library.
    """
    component_library.reset()
    _load_mot_components()
    _load_home_folder()
    _load_automatic_cascades()


def get_model(model_name):
    """Load the class of one of the available models.

    Args:
        model_name (str): One of the models from the cascade models or composite models

    Returns:
        class: Either a cascade model or a composite model. In any case, a model that can be given to the ``fit_model``
            function.
    """
    try:
        return component_library.get_component('cascade_models', model_name)
    except ValueError:
        try:
            return component_library.get_component('composite_models', model_name)
        except ValueError:
            raise ValueError('The model with the name "{}" could not be found.'.format(model_name))


def list_composite_models():
    """Get a name listing of all available composite models.

    Returns:
        list of str: a list of available composite model names
    """
    return component_library.get_component_list('composite_models')


def list_cascade_models(target_model_name=None):
    """Get a list of all available cascade models

    Args:
        target_model_name (str): if given we will only return the list of cascades that end with this composite model.

    Returns:
        list of str: A list of available cascade models
    """
    model_names = component_library.get_component_list('cascade_models')

    if target_model_name:
        cascades = []
        for name in model_names:
            meta_info = get_meta_info('cascade_models', name)
            if meta_info['target_model'] == target_model_name:
                cascades.append(name)
        return cascades

    return model_names


def get_batch_profile(batch_profile):
    """Load the class of one of the available batch profiles

    Args:
        batch_profile (str): The name of the batch profile class to load

    Returns:
        cls: the batch profile class
    """
    return component_library.get_component('batch_profiles', batch_profile)


def _load_mot_components():
    """Load all the components from MOT."""
    items = [
        (mot.library_functions, CLLibrary, 'library_functions'),
        (mdt.model_building.likelihood_functions, LikelihoodFunction, 'likelihood_functions'),
        (mdt.model_building.parameter_functions.transformations, AbstractTransformation, 'parameter_transforms'),
    ]

    for module_obj, class_type, component_type in items:
        module_items = inspect.getmembers(module_obj, lambda cls: inspect.isclass(cls) and issubclass(cls, class_type))
        for item in [x[0] for x in module_items if x[0] != class_type.__name__]:
            add_component(component_type, item, getattr(module_obj, item))


def _load_home_folder():
    """Load the components from the MDT home folder.

    This first loads all components from the ``standard`` folder and next all those from the ``user`` folder.
    """
    for user_type in ['standard', 'user']:
        base_path = os.path.join(get_config_dir(), 'components', user_type)
        for path, sub_dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    full_path = os.path.join(path, file)

                    module_name = os.path.splitext(full_path[len(os.path.join(get_config_dir(), 'components')):])[0]

                    try:
                        SourceFileLoader(module_name, full_path).load_module()
                    except Exception as e:
                        logger = logging.getLogger(__name__)
                        logger.warning('Could not load the file "{}", exception: "{}".'.format(full_path, str(e)))


def _load_automatic_cascades():
    """Automatically create cascade models where possible.

    This generates cascade models matching the scheme in Harms 2017: CS, CI and CF cascades:

    - CS: Cascade S0, a cascade which only initializes the model with an S0 estimate
    - CI: Cascade Initialized: initalizes the volume fractions and orientations
    - CF: Cascade Fixed: initializes the volume fractions, fixes the orientations
    """
    from mdt.configuration import use_automatic_generated_cascades, get_automatic_generated_cascades_excluded

    def get_missing_s0_cascades(models, cascades):
        missing_cascades = []
        for model_name in models:
            if '{} (Cascade|S0)'.format(model_name) not in cascades:
                missing_cascades.append('{} (Cascade|S0)'.format(model_name))
        return missing_cascades

    def generate_cascade(cascaded_name):
        from mdt.component_templates.cascade_models import CascadeTemplate

        if '(Cascade|S0)' in cascaded_name:
            class Template(CascadeTemplate):
                cascade_name_modifier = 'S0'
                description = 'Automatically generated cascade.'
                models = ('S0',
                          cascaded_name[0:-len('(Cascade|S0)')].strip())

    if use_automatic_generated_cascades():
        excludes = get_automatic_generated_cascades_excluded()
        models_list = [m for m in list_composite_models() if m not in excludes]

        for cascade in get_missing_s0_cascades(models_list, list_cascade_models()):
            generate_cascade(cascade)
