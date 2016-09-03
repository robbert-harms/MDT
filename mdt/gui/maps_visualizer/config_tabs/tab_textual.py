import yaml
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget

from mdt.gui.maps_visualizer.actions import NewConfigAction
from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig
from mdt.gui.maps_visualizer.design.ui_TabTextual import Ui_TabTextual
from mdt.gui.utils import blocked_signals
from mdt.visualization.maps.base import DataInfo

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

        self._flags = {'updating_config_from_string': False}

    @pyqtSlot(DataInfo)
    def set_new_data(self, data_info):
        pass

    @pyqtSlot(ValidatedMapPlotConfig)
    def set_new_config(self, config):
        with blocked_signals(self.textConfigEdit):
            if not self._flags['updating_config_from_string']:
                self.textConfigEdit.setPlainText(config.to_yaml())

    @pyqtSlot(str)
    def _config_from_string(self, text):
        self._flags['updating_config_from_string'] = True
        text = text.replace('\t', ' '*4)
        try:
            self._controller.apply_action(NewConfigAction(ValidatedMapPlotConfig.from_yaml(text)))
        except yaml.parser.ParserError:
            pass
        except yaml.scanner.ScannerError:
            pass
        except ValueError:
            pass
        finally:
            self._flags['updating_config_from_string'] = False
