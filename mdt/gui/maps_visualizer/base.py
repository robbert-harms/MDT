from PyQt5.QtCore import QObject, pyqtSignal
from mdt.visualization.maps.base import MapPlotConfig, SimpleDataInfo


class DataConfigModel(object):
    """The model contains all the state information for viewing the maps, both the data and the configuration."""

    def get_data(self):
        """Get the current data.

        Returns:
            mdt.visualization.maps.base.DataInfo: the information about the data we are viewing
        """
        raise NotImplementedError()

    def get_config(self):
        """Get the current configuration.

        Returns:
            mdt.visualization.maps.base.MapPlotConfig: the visualization configuration.
        """
        raise NotImplementedError()


class SimpleDataConfigModel(DataConfigModel):

    def __init__(self, data, config):
        """The model contains all the state information of the current viewing.

        Args:
            data (mdt.visualization.maps.base.DataInfo): the data information object
            config (mdt.visualization.maps.base.MapPlotConfig): the configuration
        """
        self._data = data
        self._config = config

    def get_data(self):
        return self._data

    def get_config(self):
        return self._config


class Controller(object):

    def __init__(self):
        """Controller interface"""
        super().__init__()

    def get_model(self):
        """Get the model the view should represent.

        Returns:
            DataConfigModel: the model the view should represent. This is also the model the actions can use for updating
                the model.
        """
        raise NotImplementedError()

    def set_data(self, data_info, config=None):
        """Set new data to visualize.

        Args:
            data_info (mdt.visualization.maps.base.DataInfo): the new data to visualize
            config (mdt.visualization.maps.base.MapPlotConfig): the new configuration for the data
                If given, we will display the new data immediately with the given config
        """

    def apply_action(self, action, store_in_history=True):
        """Apply a new configuration action.

        If there is no difference between the current config and the one generated by this new action, the
        action will not be stored in history and will not need to be applied.

        Args:
            action (mdt.gui.maps_visualizer.base.ConfigAction): the configuration action to add and apply
            store_in_history (boolean): if this action should be stored in the history or not
        """

    def undo(self):
        """Undo a previous configuration action"""

    def redo(self):
        """Reapply a previously undone configuration action"""

    def has_undo(self):
        """Check if this controller has an undo action available.

        Returns:
            boolean: True if an undo action is available.
        """

    def has_redo(self):
        """Check if this controller has an redo action available.

        Returns:
            boolean: True if an redo action is available.
        """


class QtController(Controller, QObject):

    model_updated = pyqtSignal(DataConfigModel)

    def __init__(self):
        super().__init__()
        self._current_model = SimpleDataConfigModel(SimpleDataInfo({}), MapPlotConfig())
        self._actions_history = []
        self._redoable_actions = []

    def set_data(self, data_info, config=None):
        config = config or MapPlotConfig()
        if not isinstance(config, MapPlotConfig):
            config = MapPlotConfig.from_dict(config.to_dict())

        self._current_model = SimpleDataConfigModel(data_info, config)

        if data_info.get_map_names():
            max_dim = data_info.get_max_dimension()
            if max_dim < config.dimension:
                config.dimension = max_dim

        config.maps_to_show = list(filter(lambda k: k in data_info.get_map_names(), config.maps_to_show))

        self._actions_history = []
        self._redoable_actions = []
        self.model_updated.emit(self._current_model)

    def get_model(self):
        return self._current_model

    def apply_action(self, action, store_in_history=True):
        new_model = action.apply(self._current_model)
        is_applied = self._apply_new_model(new_model)
        if is_applied:
            if store_in_history:
                self._actions_history.append(action)
                self._redoable_actions = []
            self.model_updated.emit(self._current_model)

    def undo(self):
        if len(self._actions_history):
            action = self._actions_history.pop()
            self._apply_new_model(action.unapply())
            self._redoable_actions.append(action)
            self.model_updated.emit(self._current_model)

    def redo(self):
        if len(self._redoable_actions):
            action = self._redoable_actions.pop()
            self._apply_new_model(action.apply(self._current_model))
            self._actions_history.append(action)
            self.model_updated.emit(self._current_model)

    def has_undo(self):
        return len(self._actions_history) > 0

    def has_redo(self):
        return len(self._redoable_actions) > 0

    def _apply_new_model(self, new_model):
        """Apply the current configuration.

        Args:
            new_model (DataConfigModel): the new model to apply

        Returns:
            bool: if the model was applied or not. If the difference with the current configuration
                and the old one is None, we return False. Else we return True.
        """
        new_config = new_model.get_config()
        if new_config.validate(new_model.get_data())\
                or self._current_model.get_data() != new_model.get_data():
            self._current_model = new_model
            return True
        return False
