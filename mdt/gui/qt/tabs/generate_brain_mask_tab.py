import os

import numpy as np

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from mdt import view_results_slice, load_volume, load_brain_mask, create_median_otsu_brain_mask
from mdt.gui.qt.utils import image_files_filters, protocol_files_filters
from mdt.gui.qt.design.ui_generate_brain_mask_tab import Ui_GenerateBrainMaskTabContent

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateBrainMaskTab(Ui_GenerateBrainMaskTabContent):

    def __init__(self, shared_state, computations_thread):
        self._shared_state = shared_state
        self._computations_thread = computations_thread

    def setupUi(self, tab_content):
        super(GenerateBrainMaskTab, self).setupUi(tab_content)

        self.selectImageButton.clicked.connect(lambda: self._select_image())
        self.selectProtocolButton.clicked.connect(lambda: self._select_protocol())
        self.selectOutputButton.clicked.connect(lambda: self._select_output())
        self.viewButton.clicked.connect(self.view_mask)
        self.generateButton.clicked.connect(self.generate_mask)

    @pyqtSlot()
    def _select_image(self):
        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the 4d diffusion weighted image', directory=self._shared_state.base_dir,
            filter=';;'.join(image_files_filters))

        if open_file:
            self.selectedImageText.setText(open_file)

        self._shared_state.base_dir = os.path.dirname(open_file)
        self._check_enable_action_buttons()

    @pyqtSlot()
    def _select_protocol(self):
        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the protocol', directory=self._shared_state.base_dir,
            filter=';;'.join(protocol_files_filters))

        if open_file:
            self.selectedProtocolText.setText(open_file)

        self._shared_state.base_dir = os.path.dirname(open_file)
        self._check_enable_action_buttons()

    @pyqtSlot()
    def _select_output(self):
        output_file_name, used_filter = QFileDialog().getSaveFileName(
            caption='Select the 4d diffusion weighted image', directory=self._shared_state.base_dir,
            filter=';;'.join(image_files_filters))

        if output_file_name:
            self.selectedOutputText.setText(output_file_name)
        self._check_enable_action_buttons()

    def _check_enable_action_buttons(self):
        self.generateButton.setEnabled(os.path.isfile(self.selectedImageText.text()) and
                                       os.path.isfile(self.selectedProtocolText.text()))
        self.viewButton.setEnabled(os.path.isfile(self.selectedImageText.text()) and
                                   os.path.isfile(self.selectedOutputText.text()))

    @pyqtSlot()
    def view_mask(self):
        mask = np.expand_dims(load_brain_mask(self.selectedOutputText.text()), axis=3)
        image_data = load_volume(self.selectedImageText.text())[0]
        masked_image = image_data * mask

        view_results_slice({'Masked': masked_image,
                            'DWI': image_data},
                           dimension=2,
                           slice_ind=image_data.shape[2] // 2.0)

    @pyqtSlot()
    def generate_mask(self):
        create_median_otsu_brain_mask(self.selectedImageText.text(),
                                      self.selectedProtocolText.text(),
                                      self.selectedOutputText.text(),
                                      median_radius=self.medianRadiusInput.value(),
                                      numpass=self.numberOfPassesInput.value(),
                                      mask_threshold=self.finalThresholdInput.value())
