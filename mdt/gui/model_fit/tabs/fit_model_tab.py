import os
from copy import copy
from textwrap import dedent

import yaml
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QDialogButtonBox

import mdt
import mot
from mdt.gui.maps_visualizer.main import start_gui
from mdt.lib.components import list_composite_models
from mdt.gui.model_fit.design.ui_fit_model_tab import Ui_FitModelTabContent
from mdt.gui.model_fit.design.ui_optimization_extra_data_add_protocol_map_dialog import Ui_AddProtocolMapDialog
from mdt.gui.model_fit.design.ui_optimization_extra_data_dialog import Ui_OptimizationExtraDataDialog
from mdt.gui.model_fit.design.ui_optimization_options_dialog import Ui_OptimizationOptionsDialog
from mdt.gui.utils import function_message_decorator, image_files_filters, protocol_files_filters, MainTab, \
    get_script_file_header_text
from mdt.utils import split_image_path, get_cl_devices
from mdt.visualization.maps.base import SimpleDataInfo, MapPlotConfig
from mot.optimize import get_minimizer_options

__author__ = 'Robbert Harms'
__date__ = "2016-06-27"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class FitModelTab(MainTab, Ui_FitModelTabContent, QObject):

    def __init__(self, shared_state, computations_thread):
        super().__init__()
        self._shared_state = shared_state
        self._computations_thread = computations_thread
        self._run_model_worker = RunModelWorker()
        self._tab_content = None
        self._optim_options = OptimOptions()
        self._input_data_info = InputDataInfo()

    def setupUi(self, tab_content):
        super().setupUi(tab_content)
        self._tab_content = tab_content

        self.selectDWI.clicked.connect(lambda: self._select_dwi())
        self.selectMask.clicked.connect(lambda: self._select_mask())
        self.selectProtocol.clicked.connect(lambda: self._select_protocol())
        self.selectOutputFolder.clicked.connect(lambda: self._select_output())

        self.selectedDWI.textChanged.connect(self._check_enable_action_buttons)
        self.selectedMask.textChanged.connect(self._check_enable_action_buttons)
        self.selectedProtocol.textChanged.connect(self._check_enable_action_buttons)
        self.selectedOutputFolder.textChanged.connect(self._check_enable_action_buttons)
        self.modelSelection.currentIndexChanged.connect(self._check_enable_action_buttons)

        self.runButton.clicked.connect(lambda: self._run_model())
        self.viewResultsButton.clicked.connect(lambda: self._start_maps_visualizer())
        self.optimizationOptionsButton.clicked.connect(self._run_optimization_options_dialog)
        self.additionalDataButton.clicked.connect(self._extra_data_dialog)

        self.modelSelection.addItems(list(sorted(list_composite_models())))
        initial_model = 'BallStick_r1'

        self.modelSelection.setCurrentText(initial_model)

        if self._input_data_info.dwi:
            self.selectedDWI.setText(self._input_data_info.dwi)
        if self._input_data_info.mask:
            self.selectedMask.setText(self._input_data_info.mask)
        if self._input_data_info.protocol:
            self.selectedProtocol.setText(self._input_data_info.protocol)

        self.update_output_folder_text()
        self._check_enable_action_buttons()

    def _select_dwi(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedDWI.text() != '':
            initial_dir = self.selectedDWI.text()

        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the 4d (diffusion weighted) image', directory=initial_dir,
            filter=';;'.join(image_files_filters))

        if os.path.isfile(open_file):
            self.selectedDWI.setText(open_file)
            self._input_data_info.dwi = open_file
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
            self._input_data_info.mask = open_file
            self._shared_state.base_dir = os.path.dirname(open_file)
            self.update_output_folder_text()

    def _select_protocol(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedProtocol.text() != '':
            initial_dir = self.selectedProtocol.text()

        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select the protocol', directory=initial_dir,
            filter=';;'.join(protocol_files_filters))

        if os.path.isfile(open_file):
            self.selectedProtocol.setText(open_file)
            self._input_data_info.protocol = open_file
            self._shared_state.base_dir = os.path.dirname(open_file)

    def _select_output(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedOutputFolder.text() != '':
            initial_dir = self.selectedOutputFolder.text()

        output_file_name = QFileDialog().getExistingDirectory(
            caption='Select the output folder', directory=initial_dir)

        if output_file_name:
            self.selectedOutputFolder.setText(output_file_name)

    def _check_enable_action_buttons(self):
        self.runButton.setEnabled(
            os.path.isfile(self.selectedDWI.text()) and
            os.path.isfile(self.selectedMask.text()) and
            os.path.isfile(self.selectedProtocol.text()) and
            self.selectedOutputFolder.text() != '')
        self.viewResultsButton.setEnabled(
            os.path.isdir(self.selectedOutputFolder.text() + '/' + self._get_current_model_name())
        )

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

    def _extra_data_dialog(self):
        dialog = ExtraDataDialog(self._shared_state, self._tab_content, self._input_data_info)
        return_value = dialog.exec_()

        if return_value:
            dialog.write_config()

    def _get_current_model_name(self):
        return self.modelSelection.currentText()

    def _start_maps_visualizer(self):
        folder = self.selectedOutputFolder.text() + '/' + self._get_current_model_name()
        maps = mdt.load_volume_maps(folder)
        a_map = maps[list(maps.keys())[0]]

        config = MapPlotConfig()
        config.dimension = 2

        if len(a_map.shape) > 2:
            config.slice_index = a_map.shape[2] // 2

        start_gui(data=SimpleDataInfo.from_paths([folder]), config=config, app_exec=False)

    def _run_model(self):
        model_name = self._get_current_model_name()
        model = mdt.get_model(model_name)()

        if not model.is_input_data_sufficient(self._input_data_info.build_input_data()):
            msg = ProtocolWarningBox(model.get_input_data_problems(self._input_data_info.build_input_data()))
            msg.exec_()
            return

        self._run_model_worker.set_args(
            model_name,
            self._input_data_info.build_input_data(),
            self.selectedOutputFolder.text(),
            recalculate=True,
            double_precision=self._optim_options.double_precision,
            method=self._optim_options.method,
            use_cascaded_inits=True
        )

        self._computations_thread.start()
        self._run_model_worker.moveToThread(self._computations_thread)

        self._run_model_worker.starting.connect(self._computations_thread.starting)
        self._run_model_worker.finished.connect(self._computations_thread.finished)

        self._run_model_worker.starting.connect(lambda: self.runButton.setEnabled(False))
        self._run_model_worker.finished.connect(lambda: self.runButton.setEnabled(True))
        self._run_model_worker.finished.connect(lambda: self.viewResultsButton.setEnabled(True))

        self._run_model_worker.finished.connect(
            lambda: self._shared_state.set_output_folder(self._get_full_model_output_path()))

        image_path = split_image_path(self._input_data_info.dwi)
        script_basename = os.path.join(image_path[0], 'scripts',
                                       'fit_model_{}_{}'.format(self._get_current_model_name().replace('|', '.'),
                                                                image_path[1]))
        if not os.path.isdir(os.path.join(image_path[0], 'scripts')):
            os.makedirs(os.path.join(image_path[0], 'scripts'))

        script_info = dict(optim_options=self._optim_options,
                           input_data_info=self._input_data_info,
                           model=self._get_current_model_name(),
                           output_folder=self.selectedOutputFolder.text(),
                           recalculate=True,
                           double_precision=self._optim_options.double_precision)

        self._run_model_worker.finished.connect(
            lambda: self._write_python_script_file(script_basename + '.py', **script_info))

        self._run_model_worker.finished.connect(
            lambda: self._write_bash_script_file(script_basename + '.sh', **script_info))

        self._run_model_worker.starting.emit()

    def _get_full_model_output_path(self):
        parts = [self.selectedOutputFolder.text()]
        parts.append(self._get_current_model_name().split(' ')[0])
        return os.path.join(*parts)

    def _write_python_script_file(self, output_file, **kwargs):
        input_data_info = kwargs['input_data_info']
        optim_options = kwargs['optim_options']

        all_cl_devices = get_cl_devices()
        user_selected_devices = mot.configuration.get_cl_environments()

        format_kwargs = dict(
            header=get_script_file_header_text({'Purpose': 'Fitting a model'}),
            dwi=input_data_info.dwi,
            protocol=input_data_info.protocol,
            mask=input_data_info.mask,
            noise_std=input_data_info.noise_std,
            gradient_deviations=input_data_info.gradient_deviations,
            extra_protocol=input_data_info.extra_protocol,
            model=kwargs['model'],
            output_folder=kwargs['output_folder'],
            recalculate=kwargs['recalculate'],
            double_precision=kwargs['double_precision'],
            cl_device_ind=[ind for ind, device in enumerate(all_cl_devices) if device in user_selected_devices],
            method=optim_options.method,
            patience=optim_options.patience)

        with open(output_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')

            f.write(dedent('''
                {header}

                import mdt
                
                input_data = mdt.load_input_data(
                    {dwi!r},
                    {protocol!r},
                    {mask!r},
                    noise_std={noise_std!r},
                    gradient_deviations={gradient_deviations!r},
                    extra_protocol={extra_protocol!r})
                
                mdt.fit_model(
                    {model!r},
                    input_data,
                    {output_folder!r},
                    recalculate={recalculate!r},
                    double_precision={double_precision!r},
                    cl_device_ind={cl_device_ind!r},
                    use_cascaded_inits=True,
                    method={method!r},
                    optimizer_options={{'patience': {patience!r}}})

            ''').format(**format_kwargs))

    def _write_bash_script_file(self, output_file, *args, **kwargs):
        input_data_info = kwargs['input_data_info']
        optim_options = kwargs['optim_options']

        all_cl_devices = get_cl_devices()
        user_selected_devices = mot.configuration.get_cl_environments()

        with open(output_file, 'w') as f:
            f.write('#!/usr/bin/env bash\n')
            f.write(dedent('''
                {header}

                mdt-model-fit \\
                    "{model}" \\
                    "{dwi}" \\
                    "{protocol}" \\
                    "{mask}" ''').format(header=get_script_file_header_text({'Purpose': 'Fitting a model'}),
                            model=kwargs['model'],
                            dwi=input_data_info.dwi,
                            protocol=input_data_info.protocol,
                            mask=input_data_info.mask))

            def write_new_line(line):
                f.write('\\\n' + ' ' * 4 + line + ' ')

            write_new_line('-o "{}"'.format(kwargs['output_folder']))

            if input_data_info.gradient_deviations:
                write_new_line('--gradient-deviations "{}"'.format(input_data_info.gradient_deviations))

            if input_data_info.noise_std:
                write_new_line('--noise-std {}'.format(input_data_info.noise_std))

            write_new_line('--cl-device-ind {}'.format(
                ' '.join(str(ind) for ind, device in enumerate(all_cl_devices) if device in user_selected_devices)))

            write_new_line('--recalculate' if kwargs['recalculate'] else '--no-recalculate')
            write_new_line('--double' if kwargs['double_precision'] else '--float')
            write_new_line('--method {}'.format(optim_options.method))
            write_new_line('--patience {}'.format(optim_options.patience))
            write_new_line('--use-cascaded-inits')

            if input_data_info.extra_protocol:
                write_new_line('--extra-protocol {}'.format(
                    ' '.join('{}="{}"'.format(key, value) for key, value in input_data_info.extra_protocol.items())
                ))


class ProtocolWarningBox(QMessageBox):

    def __init__(self, problems, *args):
        super().__init__(*args)
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
        super().__init__(parent)
        self._shared_state = shared_state
        self._config = config
        self.setupUi(self)

        self.patience.textChanged.connect(self._check_enable_ok_button)

        self.optimizationRoutine.addItems(sorted(OptimOptions.optim_routines.keys()))
        self.optimizationRoutine.currentIndexChanged.connect(self._update_default_patience)
        self.defaultOptimizerGroup.buttonClicked.connect(self._update_optimization_routine_selection)

        self._load_config()

    def write_config(self):
        """Write to the config the user selected options"""
        self._config.double_precision = self.doublePrecision.isChecked()
        self._config.use_model_default_optimizer = self.defaultOptimizer_True.isChecked()
        self._config.method = OptimOptions.optim_routines[self.optimizationRoutine.currentText()]
        self._config.patience = int(self.patience.text())

    def _load_config(self):
        """Load the settings from the config into the GUI"""
        self.doublePrecision.setChecked(self._config.double_precision)
        self.defaultOptimizer_False.setChecked(not self._config.use_model_default_optimizer)
        self._update_optimization_routine_selection()

        self.optimizationRoutine.setCurrentText({v: k for k, v in
                                                 OptimOptions.optim_routines.items()}[self._config.method])
        self.patience.setText(str(self._config.patience))

    def _check_enable_ok_button(self):
        enabled = True
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
        method = OptimOptions.optim_routines[self.optimizationRoutine.currentText()]
        self.patience.setText(str(get_minimizer_options(method)['patience']))


class OptimOptions:

    optim_routines = {'Powell\'s method': 'Powell',
                      'Nelder-Mead Simplex': 'Nelder-Mead',
                      'Levenberg Marquardt': 'Levenberg-Marquardt'}

    def __init__(self):
        """Storage class for communication between the options dialog and the main frame"""
        self.use_model_default_optimizer = True
        self.double_precision = False

        self.method = mdt.configuration.get_general_optimizer_name()
        self.patience = mdt.configuration.get_general_optimizer_options()['patience']

        if self.patience is None:
            self.patience = get_minimizer_options(self.method)['patience']


class InputDataInfo:

    def __init__(self):
        self.dwi = None
        self.mask = None
        self.protocol = None
        self.extra_protocol = {}
        self.gradient_deviations = None
        self.noise_std = None

    def build_input_data(self):
        return mdt.load_input_data(self.dwi, self.protocol, self.mask,
                                   noise_std=self.noise_std, gradient_deviations=self.gradient_deviations,
                                   extra_protocol=self.extra_protocol)


class ExtraDataDialog(Ui_OptimizationExtraDataDialog, QDialog):

    def __init__(self, shared_state, parent, input_data_info):
        super().__init__(parent)
        self._shared_state = shared_state
        self._input_data_info = input_data_info
        self.setupUi(self)

        self.noiseStdFileSelect.clicked.connect(lambda: self._select_std_file())
        self.noiseStd.textChanged.connect(self._check_enable_ok_button)

        self.gradDevFileSelect.clicked.connect(lambda: self._select_grad_dev_file())
        self.gradientDeviations.textChanged.connect(self._check_enable_ok_button)

        self.addProtocolMap.clicked.connect(lambda: self._add_map_dialog())
        self.removeProtocolMap.clicked.connect(lambda: self._remove_selected_extra_protocol())

        self._extra_protocol = {}

        self._load_config()

    def _add_map_dialog(self):
        dialog = AddProtocolMapDialog(self._shared_state, self, self._extra_protocol)
        return_value = dialog.exec_()

        if return_value:
            dialog.write_config()
            self._update_map_view()

    def _remove_selected_extra_protocol(self):
        to_remove = [el.text() for el in self.protocolMaps.selectedItems()]
        extra_protocol = {}
        for key, value in self._extra_protocol.items():
            if '{}: {}'.format(key, value) not in to_remove:
                extra_protocol.update({key: value})
        self._extra_protocol = extra_protocol
        self._update_map_view()

    def write_config(self):
        """Write to the config the user selected options"""
        noise_std_value = self.noiseStd.text()
        if noise_std_value == '':
            self._input_data_info.noise_std = None
        else:
            self._input_data_info.noise_std = noise_std_value
            try:
                self._input_data_info.noise_std = float(noise_std_value)
            except ValueError:
                pass

        if self.gradientDeviations.text() == '':
            self._input_data_info.gradient_deviations = None
        else:
            self._input_data_info.gradient_deviations = self.gradientDeviations.text()

        self._input_data_info.extra_protocol = copy(self._extra_protocol)

    def _load_config(self):
        """Load the settings from the config into the GUI"""
        if self._input_data_info.noise_std is not None:
            self.noiseStd.setText(str(self._input_data_info.noise_std))
        if self._input_data_info.gradient_deviations is not None:
            self.gradientDeviations.setText(str(self._input_data_info.gradient_deviations))
        self._extra_protocol = copy(self._input_data_info.extra_protocol)
        self._update_map_view()

    def _select_std_file(self):
        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select a noise std volume', directory=self._shared_state.base_dir,
            filter=';;'.join(image_files_filters))

        if open_file:
            self._shared_state.base_dir = os.path.dirname(open_file)
            self.noiseStd.setText(open_file)

    def _select_grad_dev_file(self):
        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select a gradient deviations volume', directory=self._shared_state.base_dir,
            filter=';;'.join(image_files_filters))

        if open_file:
            self._shared_state.base_dir = os.path.dirname(open_file)
            self.gradientDeviations.setText(open_file)

    def _check_enable_ok_button(self):
        noise_std_value = self.noiseStd.text()
        noise_std_value_is_float = False
        try:
            float(noise_std_value)
            noise_std_value_is_float = True
        except ValueError:
            pass

        enabled = noise_std_value == '' or noise_std_value_is_float or os.path.isfile(noise_std_value)

        if self.gradientDeviations.text() != '' and not os.path.isfile(self.gradientDeviations.text()):
            enabled = False

        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(enabled)

    def _update_map_view(self):
        self.protocolMaps.clear()
        self.protocolMaps.addItems('{}: {}'.format(key, self._extra_protocol[key])
                                   for key in sorted(self._extra_protocol))


class AddProtocolMapDialog(Ui_AddProtocolMapDialog, QDialog):

    def __init__(self, shared_state, parent, extra_protocol):
        super().__init__(parent)
        self._shared_state = shared_state
        self._extra_protocol = extra_protocol
        self.setupUi(self)

        self.mapNameInput.textChanged.connect(self._check_enable_ok_button)
        self.valueInput.textChanged.connect(self._check_enable_ok_button)

        self.fileBrowse.clicked.connect(lambda: self._select_value_file())

        self._check_enable_ok_button()

    def write_config(self):
        """Write to the config the user selected options"""
        value = self.valueInput.text()
        try:
            value = float(value)
        except ValueError:
            pass

        self._extra_protocol.update({self.mapNameInput.text(): value})

    def _select_value_file(self):
        open_file, used_filter = QFileDialog().getOpenFileName(
            caption='Select a noise std volume', directory=self._shared_state.base_dir,
            filter=';;'.join(image_files_filters))

        if open_file:
            self._shared_state.base_dir = os.path.dirname(open_file)
            self.valueInput.setText(open_file)

    def _check_enable_ok_button(self):
        value = self.valueInput.text()
        value_is_float = False
        try:
            float(value)
            value_is_float = True
        except ValueError:
            pass

        enabled = value_is_float or os.path.isfile(value)

        if self.mapNameInput.text() == '':
            enabled = False

        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(enabled)


class RunModelWorker(QObject):

    starting = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
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
