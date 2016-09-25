import os

from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QDialogButtonBox
from mdt.gui.model_fit.design.ui_optimization_options_dialog import Ui_OptimizationOptionsDialog

import mdt
from mdt.gui.model_fit.design.ui_fit_model_tab import Ui_FitModelTabContent
from mdt.gui.utils import function_message_decorator, image_files_filters, protocol_files_filters, MainTab
from mdt.utils import split_image_path
from mot.factory import get_optimizer_by_name

__author__ = 'Robbert Harms'
__date__ = "2016-06-27"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class FitModelTab(MainTab, Ui_FitModelTabContent):

    def __init__(self, shared_state, computations_thread):
        self._shared_state = shared_state
        self._computations_thread = computations_thread
        self._run_model_worker = RunModelWorker()
        self._tab_content = None
        self._optim_options = OptimOptions()

    def setupUi(self, tab_content):
        super(FitModelTab, self).setupUi(tab_content)
        self._tab_content = tab_content

        self.selectDWI.clicked.connect(lambda: self._select_dwi())
        self.selectMask.clicked.connect(lambda: self._select_mask())
        self.selectProtocol.clicked.connect(lambda: self._select_protocol())
        self.selectOutputFolder.clicked.connect(lambda: self._select_output())

        self.selectedDWI.textChanged.connect(self._check_enable_action_buttons)
        self.selectedMask.textChanged.connect(self._check_enable_action_buttons)
        self.selectedProtocol.textChanged.connect(self._check_enable_action_buttons)

        self.runButton.clicked.connect(self.run_model)
        self.optimizationOptionsButton.clicked.connect(self._run_optimization_options_dialog)

        self.modelSelection.addItems(list(sorted(mdt.get_models_list())))
        self.modelSelection.setCurrentText('BallStick (Cascade)')
        self._check_enable_action_buttons()

    def _select_dwi(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedDWI.text() != '':
            initial_dir = self.selectedDWI.text()

        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the 4d diffusion weighted image', directory=initial_dir,
            filter=';;'.join(image_files_filters))

        if os.path.isfile(open_file):
            self.selectedDWI.setText(open_file)
            self._shared_state.base_dir = os.path.dirname(open_file)
            self.update_output_folder_text()

    def _select_mask(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedMask.text() != '':
            initial_dir = self.selectedMask.text()

        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the brain mask', directory=initial_dir,
            filter=';;'.join(image_files_filters))

        if os.path.isfile(open_file):
            self.selectedMask.setText(open_file)
            self._shared_state.base_dir = os.path.dirname(open_file)
            self.update_output_folder_text()

    def _select_output(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedOutputFolder.text() != '':
            initial_dir = self.selectedOutputFolder.text()

        output_file_name = QFileDialog().getExistingDirectory(
            caption='Select the output folder', directory=initial_dir)

        if output_file_name:
            self.selectedOutputFolder.setText(output_file_name)

    def _select_protocol(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedProtocol.text() != '':
            initial_dir = self.selectedProtocol.text()

        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the protocol', directory=initial_dir,
            filter=';;'.join(protocol_files_filters))

        if os.path.isfile(open_file):
            self.selectedProtocol.setText(open_file)
            self._shared_state.base_dir = os.path.dirname(open_file)

    def _check_enable_action_buttons(self):
        self.runButton.setEnabled(
            os.path.isfile(self.selectedDWI.text()) and
            os.path.isfile(self.selectedMask.text()) and
            os.path.isfile(self.selectedProtocol.text()) and
            self.selectedOutputFolder.text() != '')

    def update_output_folder_text(self):
        if os.path.isfile(self.selectedDWI.text()) and os.path.isfile(self.selectedMask.text()):
            folder_base = os.path.join(os.path.dirname(self.selectedDWI.text()), 'output',
                                       split_image_path(self.selectedMask.text())[1])
            self.selectedOutputFolder.setText(folder_base)

    def _run_optimization_options_dialog(self):
        dialog = OptimizationOptionsDialog(self._shared_state, self._tab_content, self._optim_options)
        return_value = dialog.exec_()

        if return_value:
            dialog.write_config()

    @pyqtSlot()
    def run_model(self):
        model = mdt.get_model(self.modelSelection.currentText())
        protocol = mdt.load_protocol(self.selectedProtocol.text())

        if not model.is_protocol_sufficient(protocol):
            msg = ProtocolWarningBox(model.get_protocol_problems(protocol))
            msg.exec_()
            return

        self._run_model_worker.set_args(
            model,
            mdt.load_problem_data(self.selectedDWI.text(),
                                  self.selectedProtocol.text(),
                                  self.selectedMask.text(),
                                  noise_std=self._optim_options.noise_std),
            self.selectedOutputFolder.text(),
            recalculate=True,
            double_precision=self._optim_options.double_precision,
            only_recalculate_last=not self._optim_options.recalculate_all,
            optimizer=self._optim_options.get_optimizer(),
            save_user_script_info=False)

        self._computations_thread.start()
        self._run_model_worker.moveToThread(self._computations_thread)

        self._run_model_worker.starting.connect(self._computations_thread.starting)
        self._run_model_worker.finished.connect(self._computations_thread.finished)

        self._run_model_worker.starting.connect(lambda: self.runButton.setEnabled(False))
        self._run_model_worker.finished.connect(lambda: self.runButton.setEnabled(True))

        self._run_model_worker.finished.connect(
            lambda: self._shared_state.set_output_folder(self._get_full_model_output_path()))

        self._run_model_worker.starting.emit()

    def _get_full_model_output_path(self):
        parts = [self.selectedOutputFolder.text()]
        parts.append(self.modelSelection.currentText().split(' ')[0])
        return os.path.join(*parts)


class ProtocolWarningBox(QMessageBox):

    def __init__(self, problems, *args):
        super(ProtocolWarningBox, self).__init__(*args)
        self.setIcon(QMessageBox.Warning)
        self.setWindowTitle("Insufficient protocol")
        self.setText("The provided protocol is insufficient for this model.")
        self.setInformativeText("The reported problems are: \n{}".format('\n'.join(' - ' + str(p) for p in problems)))
        self._in_resize = False

    def resizeEvent(self, event):
        if not self._in_resize:
            self._in_resize = True
            self.setFixedWidth(self.children()[-1].size().width() + 200)
            self._in_resize = False


class OptimizationOptionsDialog(Ui_OptimizationOptionsDialog, QDialog):

    def __init__(self, shared_state, parent, config):
        super(OptimizationOptionsDialog, self).__init__(parent)
        self._shared_state = shared_state
        self._config = config
        self.setupUi(self)

        self.noiseStdFileSelect.clicked.connect(lambda: self._select_std_file())
        self.noiseStd.textChanged.connect(self._check_enable_ok_button)
        self.patience.textChanged.connect(self._check_enable_ok_button)

        self.optimizationRoutine.addItems(sorted(OptimOptions.optim_routines.keys()))
        self.optimizationRoutine.currentIndexChanged.connect(self._update_default_patience)
        self.defaultOptimizerGroup.buttonClicked.connect(self._update_optimization_routine_selection)

        self._load_config()

    def write_config(self):
        """Write to the config the user selected options"""
        noise_std_value = self.noiseStd.text()
        if noise_std_value == '':
            self._config.noise_std = None
        else:
            self._config.noise_std = noise_std_value
            try:
                self._config.noise_std = float(noise_std_value)
            except ValueError:
                pass

        self._config.double_precision = self.doublePrecision.isChecked()
        self._config.recalculate_all = self.recalculateAll_True.isChecked()
        self._config.use_model_default_optimizer = self.defaultOptimizer_True.isChecked()
        self._config.optimizer = OptimOptions.optim_routines[self.optimizationRoutine.currentText()]
        self._config.patience = int(self.patience.text())

    def _load_config(self):
        """Load the settings from the config into the GUI"""
        if self._config.noise_std is not None:
            self.noiseStd.setText(str(self._config.noise_std))
        self.doublePrecision.setChecked(self._config.double_precision)
        self.recalculateAll_True.setChecked(self._config.recalculate_all)
        self.defaultOptimizer_False.setChecked(not self._config.use_model_default_optimizer)
        self._update_optimization_routine_selection()

        self.optimizationRoutine.setCurrentText({v: k for k, v in
                                                 OptimOptions.optim_routines.items()}[self._config.optimizer])
        self.patience.setText(str(self._config.patience))

    def _select_std_file(self):
        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select a noise std volume', directory=self._shared_state.base_dir,
            filter=';;'.join(image_files_filters))

        if open_file:
            self._shared_state.base_dir = os.path.dirname(open_file)
            self.noiseStd.setText(open_file)

    def _check_enable_ok_button(self):
        noise_std_value = self.noiseStd.text()
        noise_std_value_is_float = False
        try:
            float(noise_std_value)
            noise_std_value_is_float = True
        except ValueError:
            pass

        enabled = noise_std_value == '' or noise_std_value_is_float or os.path.isfile(noise_std_value)

        if self.defaultOptimizer_False.isChecked():
            try:
                int(self.patience.text())
            except ValueError:
                enabled = False

        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(enabled)

    def _update_optimization_routine_selection(self):
        self.optimizationRoutine.setDisabled(self.defaultOptimizer_True.isChecked())
        self.patience.setDisabled(self.defaultOptimizer_True.isChecked())

    def _update_default_patience(self):
        optimizer = get_optimizer_by_name(OptimOptions.optim_routines[self.optimizationRoutine.currentText()])
        self.patience.setText(str(optimizer.default_patience))


class OptimOptions(object):

    optim_routines = {'Powell\'s method': 'Powell',
                      'Nelder-Mead Simplex': 'NMSimplex',
                      'Levenberg Marquardt': 'LevenbergMarquardt'}

    def __init__(self):
        """Storage class for communication between the options dialog and the main frame"""
        self.use_model_default_optimizer = True
        self.double_precision = False

        self.optimizer = mdt.configuration.get_optimizer_name()
        self.patience = mdt.configuration.get_optimizer_settings()['patience']

        if self.patience is None:
            self.patience = get_optimizer_by_name(self.optimizer).default_patience

        self.recalculate_all = False
        self.noise_std = None

    def get_optimizer(self):
        if self.use_model_default_optimizer:
            return None
        optimizer = get_optimizer_by_name(self.optimizer)
        return optimizer(patience=self.patience)


class RunModelWorker(QObject):

    starting = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self):
        super(RunModelWorker, self).__init__()
        self.starting.connect(self.run)
        self._args = []
        self._kwargs = {}

    def set_args(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @function_message_decorator('Starting model fitting, please wait.',
                                'Finished model fitting. You can view the results using the "View results" tab.')
    @pyqtSlot()
    def run(self):
        mdt.fit_model(*self._args, **self._kwargs)
        self.finished.emit()

