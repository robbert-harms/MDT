import os
from textwrap import dedent

from mdt.nifti import load_nifti
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from mdt.visualization.maps.base import SimpleDataInfo, MapPlotConfig
from mdt.gui.maps_visualizer.main import start_gui
from mdt.gui.model_fit.design.ui_generate_roi_mask_tab import Ui_GenerateROIMaskTabContent
from mdt.gui.utils import function_message_decorator, image_files_filters, MainTab, get_script_file_header_text
from mdt.utils import split_image_path, write_slice_roi

__author__ = 'Robbert Harms'
__date__ = "2016-06-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateROIMaskTab(MainTab, Ui_GenerateROIMaskTabContent):

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

    def _select_mask(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedMaskText.text() != '':
            initial_dir = self.selectedMaskText.text()

        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the brain mask', directory=initial_dir,
            filter=';;'.join(image_files_filters))

        if os.path.isfile(open_file):
            self.selectedMaskText.setText(open_file)
            self.mask_file_changed()
            self._shared_state.base_dir = os.path.dirname(open_file)

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

    def view_mask(self):
        data = SimpleDataInfo({'Original mask': load_nifti(self.selectedMaskText.text()).get_data(),
                               'Slice mask': load_nifti(self.selectedOutputFileText.text()).get_data()})

        config = MapPlotConfig()
        config.dimension = self.dimensionInput.value()
        config.slice_index = self.sliceInput.value()
        config.maps_to_show = ['Original mask', 'Slice mask']

        start_gui(data=data, config=config, app_exec=False)

    def generate_roi_mask(self):
        kwargs = dict(mask=self.selectedMaskText.text(), output=self.selectedOutputFileText.text(),
                      dimension=self.dimensionInput.value(), slice=self.sliceInput.value())
        self._generate_mask_worker.set_args(**kwargs)
        self._computations_thread.start()
        self._generate_mask_worker.moveToThread(self._computations_thread)

        self._generate_mask_worker.starting.connect(self._computations_thread.starting)
        self._generate_mask_worker.finished.connect(self._computations_thread.finished)

        self._generate_mask_worker.starting.connect(lambda: self.generateButton.setEnabled(False))
        self._generate_mask_worker.finished.connect(lambda: self.generateButton.setEnabled(True))
        self._generate_mask_worker.finished.connect(lambda: self.viewButton.setEnabled(True))

        image_path = split_image_path(self.selectedOutputFileText.text())
        script_basename = os.path.join(image_path[0], 'scripts', 'create_roi_mask_' + image_path[1])
        if not os.path.isdir(os.path.join(image_path[0], 'scripts')):
            os.makedirs(os.path.join(image_path[0], 'scripts'))

        self._generate_mask_worker.finished.connect(
            lambda: self._write_python_script_file(script_basename + '.py', **kwargs))

        self._generate_mask_worker.finished.connect(
            lambda: self._write_bash_script_file(script_basename + '.sh', **kwargs))

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
            self.sliceInput.setValue(dimension_max // 2.0)
            self.maxSliceLabel.setText(str(dimension_max))
        else:
            self.sliceInput.setValue(0)
            self.maxSliceLabel.setText('x')

    def update_output_file_text(self):
        if os.path.isfile(self.selectedMaskText.text()):
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

    def _write_python_script_file(self, output_file, **kwargs):
        with open(output_file, 'w') as f:
            f.write('#!/usr/bin/env python')
            f.write(dedent('''

                {header}

                import mdt

                mdt.write_slice_roi(
                    {mask!r},
                    {dimension!r},
                    {slice!r},
                    {output!r},
                    overwrite_if_exists=True)

            ''').format(header=get_script_file_header_text({'Purpose': 'Generated a slice ROI mask'}),
                        **kwargs))

    def _write_bash_script_file(self, output_file, **kwargs):
        with open(output_file, 'w') as f:
            f.write('#!/usr/bin/env bash')
            f.write(dedent('''

                {header}

                mdt-generate-roi-slice "{mask}" -d {dimension} -s {slice} -o "{output}"
            ''').format(header=get_script_file_header_text({'Purpose': 'Generated a slice ROI mask'}),
                        **kwargs))


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
    def run(self):
        write_slice_roi(self._kwargs['mask'], self._kwargs['dimension'],
                        self._kwargs['slice'], self._kwargs['output'], overwrite_if_exists=True)
        self.finished.emit()
