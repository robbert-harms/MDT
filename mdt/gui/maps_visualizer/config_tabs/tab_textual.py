from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget

from mdt.gui.maps_visualizer.actions import NewConfigAction
from mdt.gui.maps_visualizer.base import DataConfigModel
from mdt.gui.maps_visualizer.design.ui_TabTextual import Ui_TabTextual
from mdt.gui.utils import blocked_signals
from mdt.visualization.maps.base import MapPlotConfig

__author__ = 'Robbert Harms'
__date__ = "2016-09-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TabTextual(QWidget, Ui_TabTextual):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.model_updated.connect(self.set_new_model)

        self.textConfigEdit.new_config.connect(self._config_from_string)
        self._update_status_indication(True)

        self.viewSelectedOptions.clicked.connect(self._toggle_viewed_options)

        self._flags = {'updating_config_from_string': False,
                       'toggling_view_selection': False}

    @pyqtSlot(DataConfigModel)
    def set_new_model(self, model):
        with blocked_signals(self.textConfigEdit):
            if not self._flags['updating_config_from_string']:
                non_default_only = self.viewSelectedOptions.isChecked()
                self.textConfigEdit.setPlainText(model.get_config().to_yaml(non_default_only=non_default_only))
                self._update_status_indication(True)

    @pyqtSlot()
    def _toggle_viewed_options(self):
        self._flags['toggling_view_selection'] = True
        non_default_only = self.viewSelectedOptions.isChecked()
        current_model = self._controller.get_model()
        self.textConfigEdit.setPlainText(current_model.get_config().to_yaml(non_default_only=non_default_only))
        self._update_status_indication(True)

    @pyqtSlot(str)
    def _config_from_string(self, text):
        if self._flags['toggling_view_selection']:
            self._flags['toggling_view_selection'] = False
            return

        self._flags['updating_config_from_string'] = True
        text = text.replace('\t', ' '*4)
        try:
            if text.strip() != '':
                current_model = self._controller.get_model()
                new_config = MapPlotConfig.from_yaml(text)
                new_config.validate(current_model.get_data())

                self._controller.apply_action(NewConfigAction(new_config))
                self._update_status_indication(True)

        except Exception as exc:
            self._update_status_indication(False, str(exc))
            pass
        finally:
            self._flags['updating_config_from_string'] = False

    def _update_status_indication(self, is_valid, status_message=''):
        border_color = 'red'
        if is_valid:
            border_color = 'green'
        self.textConfigEdit.setStyleSheet('border: 1px solid {}'.format(border_color))
        self.correctness_label.setText(status_message)
