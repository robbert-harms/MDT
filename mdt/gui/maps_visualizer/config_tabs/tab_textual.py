import yaml
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget

from mdt.gui.maps_visualizer.actions import NewConfigAction
from mdt.gui.maps_visualizer.design.ui_TabTextual import Ui_TabTextual
from mdt.gui.utils import blocked_signals
from mdt.visualization.maps.base import DataInfo, MapPlotConfig

__author__ = 'Robbert Harms'
__date__ = "2016-09-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TabTextual(QWidget, Ui_TabTextual):

    def __init__(self, controller, parent=None):
        super(TabTextual, self).__init__(parent)
        self.setupUi(self)

        self._controller = controller
        self._controller.new_data.connect(self.set_new_data)
        self._controller.new_config.connect(self.set_new_config)

        self.textConfigEdit.new_config.connect(self._config_from_string)
        self._update_status_indication(True)

        self._flags = {'updating_config_from_string': False}

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        pass

    @pyqtSlot(MapPlotConfig)
    def set_new_config(self, config):
        with blocked_signals(self.textConfigEdit):
            if not self._flags['updating_config_from_string']:
                self.textConfigEdit.setPlainText(config.to_yaml())
                self._update_status_indication(True)

    @pyqtSlot(str)
    def _config_from_string(self, text):
        self._flags['updating_config_from_string'] = True
        text = text.replace('\t', ' '*4)
        try:
            if text.strip() != '':
                new_config = MapPlotConfig.from_yaml(text)
                new_config.validate(self._controller.get_data())

                self._controller.apply_action(NewConfigAction(new_config))
                self._update_status_indication(True)

        except Exception as exc:
            self._update_status_indication(False, str(exc))
            pass
        finally:
            self._flags['updating_config_from_string'] = False

    def _update_status_indication(self, status, status_message=''):
        if status:
            self.textConfigEdit.setStyleSheet("border: 1px solid green")
        else:
            self.textConfigEdit.setStyleSheet("border: 1px solid red")
        self.correctness_label.setText(status_message)
