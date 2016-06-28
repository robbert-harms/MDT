import os

from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from mdt import view_results_slice, load_volume, create_slice_roi, load_nifti
from mdt.gui.qt.design.ui_generate_roi_mask_tab import Ui_GenerateROIMaskTabContent
from mdt.gui.qt.utils import image_files_filters
from mdt.gui.utils import function_message_decorator
from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateROIMaskTab(Ui_GenerateROIMaskTabContent):

    def __init__(self, shared_state, computations_thread):
        self._shared_state = shared_state
        self._computations_thread = computations_thread
        self._generate_mask_worker = GenerateROIMaskWorker()

    def setupUi(self, tab_content):
        super(GenerateROIMaskTab, self).setupUi(tab_content)

        self.selectMaskButton.clicked.connect(lambda: self._select_mask())
        self.selectOutputFileInput.clicked.connect(lambda: self._select_output_file())

        self.viewButton.clicked.connect(self.view_mask)
        self.generateButton.clicked.connect(self.generate_roi_mask)

        self.selectedMaskText.textChanged.connect(self._check_enable_action_buttons)
        self.selectedOutputFileText.textChanged.connect(self._check_enable_action_buttons)
        self.selectedMaskText.textChanged.connect(self.mask_file_changed)

        self.dimensionInput.valueChanged.connect(self.update_dimension)
        self.sliceInput.valueChanged.connect(self.update_slice_index)

    @pyqtSlot()
    def _select_mask(self):
        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the brain mask', directory=self._shared_state.base_dir,
            filter=';;'.join(image_files_filters))

        if open_file:
            self.selectedMaskText.setText(open_file)
            self.mask_file_changed()
            self._shared_state.base_dir = os.path.dirname(open_file)

    @pyqtSlot()
    def _select_output_file(self):
        output_file_name, used_filter = QFileDialog().getSaveFileName(
            caption='Select the output file', directory=self._shared_state.base_dir,
            filter=';;'.join(image_files_filters))

        if output_file_name:
            self.selectedOutputFileText.setText(output_file_name)

    def _check_enable_action_buttons(self):
        self.generateButton.setEnabled(os.path.isfile(self.selectedMaskText.text()) and
                                       os.path.isdir(os.path.dirname(self.selectedOutputFileText.text())))
        self.viewButton.setEnabled(os.path.isfile(self.selectedMaskText.text()) and
                                   os.path.isfile(self.selectedOutputFileText.text()))

    @pyqtSlot()
    def view_mask(self):
        view_results_slice({'Original mask': load_volume(self.selectedMaskText.text())[0],
                            'Slice mask': load_volume(self.selectedOutputFileText.text())[0]},
                           dimension=self.dimensionInput.value(),
                           slice_ind=self.sliceInput.value())

    @pyqtSlot()
    def generate_roi_mask(self):
        self._generate_mask_worker.set_args(mask=self.selectedMaskText.text(),
                                            output=self.selectedOutputFileText.text(),
                                            dimension=self.dimensionInput.value(),
                                            slice=self.sliceInput.value())
        self._computations_thread.start()
        self._generate_mask_worker.moveToThread(self._computations_thread)

        self._generate_mask_worker.starting.connect(self._computations_thread.starting)
        self._generate_mask_worker.finished.connect(self._computations_thread.finished)

        self._generate_mask_worker.starting.connect(lambda: self.generateButton.setEnabled(False))
        self._generate_mask_worker.finished.connect(lambda: self.generateButton.setEnabled(True))
        self._generate_mask_worker.finished.connect(lambda: self.viewButton.setEnabled(True))

        self._generate_mask_worker.starting.emit()

    @pyqtSlot(int)
    def update_dimension(self, value):
        if os.path.isfile(self.selectedMaskText.text()):
            self.update_slice_selector()
            self.update_output_file_text()

    @pyqtSlot(int)
    def update_slice_index(self, value):
        self.update_output_file_text()

    def mask_file_changed(self):
        self.dimensionInput.setValue(2)
        self.update_slice_selector()
        self.update_output_file_text()

    def update_slice_selector(self):
        if os.path.isfile(self.selectedMaskText.text()):
            dimension_max = load_nifti(self.selectedMaskText.text()).shape[self.dimensionInput.value()]
            self.sliceInput.setMaximum(dimension_max)
            self.sliceInput.setValue(0)
            self.maxSliceLabel.setText('/ {}'.format(dimension_max))
        else:
            self.sliceInput.setValue(0)
            self.maxSliceLabel.setText('/ x')

    def update_output_file_text(self):
        folder, basename, ext = split_image_path(self.selectedMaskText.text())
        folder_base = os.path.join(folder, basename)

        if self.selectedOutputFileText.text() == '':
            self.selectedOutputFileText.setText('{}_{}_{}.nii.gz'.format(folder_base,
                                                                         self.dimensionInput.value(),
                                                                         self.sliceInput.value()))
        elif self.selectedOutputFileText.text()[0:len(folder_base)] == folder_base:
            self.selectedOutputFileText.setText('{}_{}_{}.nii.gz'.format(folder_base,
                                                                         self.dimensionInput.value(),
                                                                         self.sliceInput.value()))


class GenerateROIMaskWorker(QObject):

    starting = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self):
        super(GenerateROIMaskWorker, self).__init__()
        self.starting.connect(self.run)
        self._args = []
        self._kwargs = {}

    def set_args(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @function_message_decorator('Started with generating a slice ROI', 'Finished generating a slice ROI')
    @pyqtSlot()
    def run(self):
        create_slice_roi(self._kwargs['mask'], self._kwargs['dimension'],
                         self._kwargs['slice'], self._kwargs['output'], overwrite_if_exists=True)
        self.finished.emit()
