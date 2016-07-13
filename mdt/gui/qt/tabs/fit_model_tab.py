import os

from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

import mdt
from mdt.gui.qt.design.ui_fit_model_tab import Ui_FitModelTabContent
from mdt.gui.qt.utils import image_files_filters, protocol_files_filters, MainTab
from mdt.gui.utils import function_message_decorator

from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2016-06-27"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class FitModelTab(MainTab, Ui_FitModelTabContent):

    def __init__(self, shared_state, computations_thread):
        self._shared_state = shared_state
        self._computations_thread = computations_thread
        self._run_model_worker = RunModelWorker()

    def setupUi(self, tab_content):
        super(FitModelTab, self).setupUi(tab_content)

        self.selectDWI.clicked.connect(lambda: self._select_dwi())
        self.selectMask.clicked.connect(lambda: self._select_mask())
        self.selectProtocol.clicked.connect(lambda: self._select_protocol())
        self.selectOutputFolder.clicked.connect(lambda: self._select_output())

        self.selectedDWI.textChanged.connect(self._check_enable_action_buttons)
        self.selectedMask.textChanged.connect(self._check_enable_action_buttons)
        self.selectedProtocol.textChanged.connect(self._check_enable_action_buttons)

        self.runButton.clicked.connect(self.run_model)

        self.modelSelection.addItems(list(sorted(mdt.get_models_list())))
        self.modelSelection.setCurrentText('BallStick (Cascade)')

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

    @pyqtSlot()
    def run_model(self):
        self._run_model_worker.set_args(
            self.modelSelection.currentText(),
            mdt.load_problem_data(self.selectedDWI.text(),
                                  self.selectedProtocol.text(),
                                  self.selectedMask.text()),
            self.selectedOutputFolder.text(),
            recalculate=True)

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
