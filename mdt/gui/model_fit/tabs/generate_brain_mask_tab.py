import os
from textwrap import dedent

from mdt.nifti import load_nifti
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from mdt import load_brain_mask, create_median_otsu_brain_mask
from mdt.utils import split_image_path
from mdt.visualization.maps.base import SimpleDataInfo, MapPlotConfig
from mdt.gui.maps_visualizer.main import start_gui
from mdt.gui.model_fit.design.ui_generate_brain_mask_tab import Ui_GenerateBrainMaskTabContent
from mdt.gui.utils import function_message_decorator, image_files_filters, protocol_files_filters, MainTab, \
    get_script_file_header_text

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

            if self.selectedOutputText.text() == '':
                split_path = split_image_path(open_file)
                self.selectedOutputText.setText(os.path.join(split_path[0], split_path[1] + '_mask' + split_path[2]))

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

    def view_mask(self):
        mask = np.expand_dims(load_brain_mask(self.selectedOutputText.text()), axis=3)
        image_data = load_nifti(self.selectedImageText.text()).get_data()
        masked_image = image_data * mask

        data = SimpleDataInfo({'Masked': masked_image, 'DWI': image_data})

        config = MapPlotConfig()
        config.dimension = 2
        config.slice_index = image_data.shape[2] // 2
        config.maps_to_show = ['DWI', 'Masked']

        start_gui(data=data, config=config, app_exec=False)

    def generate_mask(self):
        args = (self.selectedImageText.text(), self.selectedProtocolText.text(), self.selectedOutputText.text())
        kwargs = dict(median_radius=self.medianRadiusInput.value(), numpass=self.numberOfPassesInput.value(),
                      mask_threshold=self.finalThresholdInput.value())

        self._generate_mask_worker.set_args(*args, **kwargs)
        self._computations_thread.start()
        self._generate_mask_worker.moveToThread(self._computations_thread)

        self._generate_mask_worker.starting.connect(self._computations_thread.starting)
        self._generate_mask_worker.finished.connect(self._computations_thread.finished)

        self._generate_mask_worker.starting.connect(lambda: self.generateButton.setEnabled(False))
        self._generate_mask_worker.finished.connect(lambda: self.generateButton.setEnabled(True))
        self._generate_mask_worker.finished.connect(lambda: self.viewButton.setEnabled(True))

        image_path = split_image_path(self.selectedOutputText.text())
        script_basename = os.path.join(image_path[0], 'scripts', 'generate_mask_' + image_path[1])
        if not os.path.isdir(os.path.join(image_path[0], 'scripts')):
            os.makedirs(os.path.join(image_path[0], 'scripts'))

        self._generate_mask_worker.finished.connect(
            lambda: self._write_python_script_file(script_basename + '.py', *args, **kwargs))

        self._generate_mask_worker.finished.connect(
            lambda: self._write_bash_script_file(script_basename + '.sh', *args, **kwargs))

        self._generate_mask_worker.starting.emit()

    def _write_python_script_file(self, output_file, *args, **kwargs):
        with open(output_file, 'w') as f:
            f.write('#!/usr/bin/env python')
            f.write(dedent('''

                {header}

                import mdt

                mdt.create_median_otsu_brain_mask(
                    {args},
                    {kwargs})

            ''').format(header=get_script_file_header_text({'Purpose': 'Generated a brain mask'}),
                        args=', \n\t'.join("{!r}".format(arg) for arg in args),
                        kwargs=', \n\t'.join('{}={!r}'.format(*el) for el in kwargs.items())))

    def _write_bash_script_file(self, output_file, *args, **kwargs):
        with open(output_file, 'w') as f:
            f.write('#!/usr/bin/env bash')
            f.write(dedent('''

                {header}

                mdt-generate-mask "{data}" "{prtcl}" -o "{output}" {kwargs}
            ''').format(header=get_script_file_header_text({'Purpose': 'Generated a brain mask'}),
                        data=args[0], prtcl=args[1], output=args[2],
                        kwargs=' '.join('--{} {!r}'.format(el[0].replace('_', '-'), el[1]) for el in kwargs.items())))


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
    def run(self):
        create_median_otsu_brain_mask(*self._args, **self._kwargs)
        self.finished.emit()
