import os

import nibabel as nib
import numpy as np
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from mdt import load_brain_mask, create_median_otsu_brain_mask
from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig
from mdt.visualization.maps.base import DataInfo
from mdt.gui.maps_visualizer.main import start_gui
from mdt.gui.model_fit.design.ui_generate_brain_mask_tab import Ui_GenerateBrainMaskTabContent
from mdt.gui.utils import function_message_decorator, image_files_filters, protocol_files_filters, MainTab

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateBrainMaskTab(MainTab, Ui_GenerateBrainMaskTabContent):

    def __init__(self, shared_state, computations_thread):
        self._shared_state = shared_state
        self._computations_thread = computations_thread
        self._generate_mask_worker = GenerateMaskWorker()

    def setupUi(self, tab_content):
        super(GenerateBrainMaskTab, self).setupUi(tab_content)

        self.selectImageButton.clicked.connect(lambda: self._select_image())
        self.selectProtocolButton.clicked.connect(lambda: self._select_protocol())
        self.selectOutputButton.clicked.connect(lambda: self._select_output())
        self.viewButton.clicked.connect(self.view_mask)
        self.generateButton.clicked.connect(self.generate_mask)

        self.selectedImageText.textChanged.connect(self._check_enable_action_buttons)
        self.selectedOutputText.textChanged.connect(self._check_enable_action_buttons)
        self.selectedProtocolText.textChanged.connect(self._check_enable_action_buttons)

    def _select_image(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedImageText.text() != '':
            initial_dir = self.selectedImageText.text()

        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the 4d diffusion weighted image', directory=initial_dir,
            filter=';;'.join(image_files_filters))

        if os.path.isfile(open_file):
            self.selectedImageText.setText(open_file)
            self._shared_state.base_dir = os.path.dirname(open_file)

    def _select_protocol(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedProtocolText.text() != '':
            initial_dir = self.selectedProtocolText.text()

        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the protocol', directory=initial_dir,
            filter=';;'.join(protocol_files_filters))

        if os.path.isfile(open_file):
            self.selectedProtocolText.setText(open_file)
            self._shared_state.base_dir = os.path.dirname(open_file)

    def _select_output(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedOutputText.text() != '':
            initial_dir = self.selectedOutputText.text()

        output_file_name, used_filter = QFileDialog().getSaveFileName(
            caption='Select the output file', directory=initial_dir,
            filter=';;'.join(image_files_filters))

        if output_file_name:
            self.selectedOutputText.setText(output_file_name)

    def _check_enable_action_buttons(self):
        self.generateButton.setEnabled(os.path.isfile(self.selectedImageText.text()) and
                                       os.path.isfile(self.selectedProtocolText.text())
                                       and os.path.isdir(os.path.dirname(self.selectedOutputText.text())))
        self.viewButton.setEnabled(os.path.isfile(self.selectedImageText.text()) and
                                   os.path.isfile(self.selectedOutputText.text()))

    @pyqtSlot()
    def view_mask(self):
        mask = np.expand_dims(load_brain_mask(self.selectedOutputText.text()), axis=3)
        image_data = nib.load(self.selectedImageText.text()).get_data()
        masked_image = image_data * mask

        data = DataInfo({'Masked': masked_image, 'DWI': image_data})
        data.directory = os.path.dirname(self.selectedImageText.text())

        config = ValidatedMapPlotConfig()
        config.dimension = 2
        config.slice_index = image_data.shape[2] // 2.0
        config.maps_to_show = ['DWI', 'Masked']

        start_gui(data=data, config=config, app_exec=False)

    @pyqtSlot()
    def generate_mask(self):
        self._generate_mask_worker.set_args(self.selectedImageText.text(),
                                            self.selectedProtocolText.text(),
                                            self.selectedOutputText.text(),
                                            median_radius=self.medianRadiusInput.value(),
                                            numpass=self.numberOfPassesInput.value(),
                                            mask_threshold=self.finalThresholdInput.value())
        self._computations_thread.start()
        self._generate_mask_worker.moveToThread(self._computations_thread)

        self._generate_mask_worker.starting.connect(self._computations_thread.starting)
        self._generate_mask_worker.finished.connect(self._computations_thread.finished)

        self._generate_mask_worker.starting.connect(lambda: self.generateButton.setEnabled(False))
        self._generate_mask_worker.finished.connect(lambda: self.generateButton.setEnabled(True))
        self._generate_mask_worker.finished.connect(lambda: self.viewButton.setEnabled(True))

        self._generate_mask_worker.starting.emit()


class GenerateMaskWorker(QObject):

    starting = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self):
        super(GenerateMaskWorker, self).__init__()
        self.starting.connect(self.run)
        self._args = []
        self._kwargs = {}

    def set_args(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @function_message_decorator('Started creating a mask.', 'Finished creating a mask.')
    @pyqtSlot()
    def run(self):
        create_median_otsu_brain_mask(*self._args, **self._kwargs)
        self.finished.emit()
