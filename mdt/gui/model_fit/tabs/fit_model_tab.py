import os
from copy import copy
from textwrap import dedent

import yaml
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QDialogButtonBox

import mdt
import mot
from mdt.gui.model_fit.design.ui_fit_model_tab import Ui_FitModelTabContent
from mdt.gui.model_fit.design.ui_optimization_extra_data_add_static_map_dialog import Ui_AddStaticMapDialog
from mdt.gui.model_fit.design.ui_optimization_extra_data_dialog import Ui_OptimizationExtraDataDialog
from mdt.gui.model_fit.design.ui_optimization_options_dialog import Ui_OptimizationOptionsDialog
from mdt.gui.utils import function_message_decorator, image_files_filters, protocol_files_filters, MainTab, \
    get_script_file_header_text
from mdt.utils import split_image_path
from mot.cl_environments import CLEnvironmentFactory
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
        self._problem_data_info = ProblemDataInfo()

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

        self.runButton.clicked.connect(lambda: self.run_model())
        self.optimizationOptionsButton.clicked.connect(self._run_optimization_options_dialog)
        self.additionalDataButton.clicked.connect(self._extra_data_dialog)

        self.modelSelection.addItems(list(sorted(mdt.get_models_list())))
        self.modelSelection.setCurrentText('BallStick_r1 (Cascade)')

        if self._problem_data_info.dwi:
            self.selectedDWI.setText(self._problem_data_info.dwi)
        if self._problem_data_info.mask:
            self.selectedMask.setText(self._problem_data_info.mask)
        if self._problem_data_info.protocol:
            self.selectedProtocol.setText(self._problem_data_info.protocol)

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
            self._problem_data_info.dwi = open_file
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
            self._problem_data_info.mask = open_file
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
            self._problem_data_info.protocol = open_file
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
        dialog = ExtraDataDialog(self._shared_state, self._tab_content, self._problem_data_info)
        return_value = dialog.exec_()

        if return_value:
            dialog.write_config()

    @pyqtSlot()
    def run_model(self):
        model = mdt.get_model(self.modelSelection.currentText())
        protocol = mdt.load_protocol(self._problem_data_info.protocol)

        if not model.is_protocol_sufficient(protocol):
            msg = ProtocolWarningBox(model.get_protocol_problems(protocol))
            msg.exec_()
            return

        self._run_model_worker.set_args(
            model,
            self._problem_data_info.get_problem_data(),
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

        image_path = split_image_path(self._problem_data_info.dwi)
        script_basename = os.path.join(image_path[0], 'scripts',
                                       'fit_model_{}_{}'.format(self.modelSelection.currentText().replace('|', '.'),
                                                                image_path[1]))
        if not os.path.isdir(os.path.join(image_path[0], 'scripts')):
            os.makedirs(os.path.join(image_path[0], 'scripts'))

        script_info = dict(optim_options=self._optim_options,
                           problem_data_info=self._problem_data_info,
                           model=self.modelSelection.currentText(),
                           output_folder=self.selectedOutputFolder.text(),
                           recalculate=True,
                           double_precision=self._optim_options.double_precision,
                           only_recalculate_last=not self._optim_options.recalculate_all,
                           save_user_script_info=False)

        self._run_model_worker.finished.connect(
            lambda: self._write_python_script_file(script_basename + '.py', **script_info))

        self._run_model_worker.finished.connect(
            lambda: self._write_bash_script_file(script_basename + '.sh', **script_info))

        self._run_model_worker.starting.emit()

    def _get_full_model_output_path(self):
        parts = [self.selectedOutputFolder.text()]
        parts.append(self.modelSelection.currentText().split(' ')[0])
        return os.path.join(*parts)

    def _write_python_script_file(self, output_file, **kwargs):
        problem_data_info = kwargs['problem_data_info']
        optim_options = kwargs['optim_options']

        all_cl_devices = CLEnvironmentFactory.smart_device_selection()
        user_selected_devices = mot.configuration.get_cl_environments()

        format_kwargs = dict(
            header=get_script_file_header_text({'Purpose': 'Fitting a model'}),
            dwi=problem_data_info.dwi,
            protocol=problem_data_info.protocol,
            mask=problem_data_info.mask,
            noise_std=problem_data_info.noise_std,
            gradient_deviations=problem_data_info.gradient_deviations,
            static_maps=problem_data_info.static_maps,
            model=kwargs['model'],
            output_folder=kwargs['output_folder'],
            recalculate=kwargs['recalculate'],
            only_recalculate_last=kwargs['only_recalculate_last'],
            double_precision=kwargs['double_precision'],
            cl_device_ind=[ind for ind, device in enumerate(all_cl_devices) if device in user_selected_devices])

        with open(output_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')

            if optim_options.use_model_default_optimizer:
                f.write(dedent('''
                    {header}

                    import mdt

                    problem_data = mdt.load_problem_data(
                        {dwi!r},
                        {protocol!r},
                        {mask!r},
                        noise_std={noise_std!r},
                        gradient_deviations={gradient_deviations!r},
                        static_maps={static_maps!r})

                    mdt.fit_model(
                        {model!r},
                        problem_data,
                        {output_folder!r},
                        recalculate={recalculate!r},
                        only_recalculate_last={only_recalculate_last!r},
                        double_precision={double_precision!r},
                        cl_device_ind={cl_device_ind!r})

                ''').format(**format_kwargs))
            else:
                format_kwargs.update({'optimizer': optim_options.optimizer, 'patience': optim_options.patience})
                f.write(dedent('''
                    {header}

                    import mdt
                    from mdt.configuration import SetGeneralOptimizer

                    problem_data = mdt.load_problem_data(
                        {dwi!r},
                        {protocol!r},
                        {mask!r},
                        noise_std={noise_std!r},
                        gradient_deviations={gradient_deviations!r},
                        static_maps={static_maps!r})

                    with mdt.config_context(SetGeneralOptimizer({optimizer!r}, settings={{'patience': {patience!r}}})):
                        mdt.fit_model(
                            {model!r},
                            problem_data,
                            {output_folder!r},
                            recalculate={recalculate!r},
                            only_recalculate_last={only_recalculate_last!r},
                            double_precision={double_precision!r},
                            cl_device_ind={cl_device_ind!r})

                ''').format(**format_kwargs))

    def _write_bash_script_file(self, output_file, *args, **kwargs):
        problem_data_info = kwargs['problem_data_info']
        optim_options = kwargs['optim_options']

        all_cl_devices = CLEnvironmentFactory.smart_device_selection()
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
                            dwi=problem_data_info.dwi,
                            protocol=problem_data_info.protocol,
                            mask=problem_data_info.mask))

            def write_new_line(line):
                f.write('\\\n' + ' ' * 4 + line + ' ')

            write_new_line('-o "{}"'.format(kwargs['output_folder']))

            if problem_data_info.gradient_deviations:
                write_new_line('--gradient-deviations "{}"'.format(problem_data_info.gradient_deviations))

            if problem_data_info.noise_std:
                write_new_line('--noise-std {}'.format(problem_data_info.noise_std))

            write_new_line('--cl-device-ind {}'.format(
                ' '.join(str(ind) for ind, device in enumerate(all_cl_devices) if device in user_selected_devices)))

            write_new_line('--recalculate' if kwargs['recalculate'] else '--no-recalculate')
            write_new_line('--only-recalculate-last' if kwargs['only_recalculate_last'] else '--recalculate-all')
            write_new_line('--double' if kwargs['double_precision'] else '--float')

            if problem_data_info.static_maps:
                write_new_line('--static-maps {}'.format(
                    ' '.join('{}="{}"'.format(key, value) for key, value in problem_data_info.static_maps.items())
                ))

            if not optim_options.use_model_default_optimizer:
                config = '''
                    optimization:
                        general:
                            name: {}
                            settings:
                                patience: {}
                '''.format(optim_options.optimizer, optim_options.patience)
                config_context = yaml.safe_dump(yaml.safe_load(config), default_flow_style=True).rstrip()
                write_new_line('--config-context "{}"'.format(config_context))


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

        self.patience.textChanged.connect(self._check_enable_ok_button)

        self.optimizationRoutine.addItems(sorted(OptimOptions.optim_routines.keys()))
        self.optimizationRoutine.currentIndexChanged.connect(self._update_default_patience)
        self.defaultOptimizerGroup.buttonClicked.connect(self._update_optimization_routine_selection)

        self._load_config()

    def write_config(self):
        """Write to the config the user selected options"""
        self._config.double_precision = self.doublePrecision.isChecked()
        self._config.recalculate_all = self.recalculateAll_True.isChecked()
        self._config.use_model_default_optimizer = self.defaultOptimizer_True.isChecked()
        self._config.optimizer = OptimOptions.optim_routines[self.optimizationRoutine.currentText()]
        self._config.patience = int(self.patience.text())

    def _load_config(self):
        """Load the settings from the config into the GUI"""
        self.doublePrecision.setChecked(self._config.double_precision)
        self.recalculateAll_True.setChecked(self._config.recalculate_all)
        self.defaultOptimizer_False.setChecked(not self._config.use_model_default_optimizer)
        self._update_optimization_routine_selection()

        self.optimizationRoutine.setCurrentText({v: k for k, v in
                                                 OptimOptions.optim_routines.items()}[self._config.optimizer])
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

        self.optimizer = mdt.configuration.get_general_optimizer_name()
        self.patience = mdt.configuration.get_general_optimizer_settings()['patience']

        if self.patience is None:
            self.patience = get_optimizer_by_name(self.optimizer).default_patience

        self.recalculate_all = False

    def get_optimizer(self):
        if self.use_model_default_optimizer:
            return None
        optimizer = get_optimizer_by_name(self.optimizer)
        return optimizer(patience=self.patience)


class ProblemDataInfo(object):

    def __init__(self):
        self.dwi = None
        self.mask = None
        self.protocol = None
        self.static_maps = {}
        self.gradient_deviations = None
        self.noise_std = None

    def get_problem_data(self):
        return mdt.load_problem_data(self.dwi, self.protocol, self.mask,
                                     noise_std=self.noise_std, gradient_deviations=self.gradient_deviations,
                                     static_maps=self.static_maps)


class ExtraDataDialog(Ui_OptimizationExtraDataDialog, QDialog):

    def __init__(self, shared_state, parent, problem_data_info):
        super(ExtraDataDialog, self).__init__(parent)
        self._shared_state = shared_state
        self._problem_data_info = problem_data_info
        self.setupUi(self)

        self.noiseStdFileSelect.clicked.connect(lambda: self._select_std_file())
        self.noiseStd.textChanged.connect(self._check_enable_ok_button)

        self.gradDevFileSelect.clicked.connect(lambda: self._select_grad_dev_file())
        self.gradientDeviations.textChanged.connect(self._check_enable_ok_button)

        self.addStaticMap.clicked.connect(lambda: self._add_map_dialog())
        self.removeStaticMap.clicked.connect(lambda: self._remove_selected_static_maps())

        self._static_maps = {}

        self._load_config()

    def _add_map_dialog(self):
        dialog = AddStaticMapDialog(self._shared_state, self, self._static_maps)
        return_value = dialog.exec_()

        if return_value:
            dialog.write_config()
            self._update_map_view()

    def _remove_selected_static_maps(self):
        to_remove = [el.text() for el in self.staticMaps.selectedItems()]
        static_maps = {}
        for key, value in self._static_maps.items():
            if '{}: {}'.format(key, value) not in to_remove:
                static_maps.update({key: value})
        self._static_maps = static_maps
        self._update_map_view()

    def write_config(self):
        """Write to the config the user selected options"""
        noise_std_value = self.noiseStd.text()
        if noise_std_value == '':
            self._problem_data_info.noise_std = None
        else:
            self._problem_data_info.noise_std = noise_std_value
            try:
                self._problem_data_info.noise_std = float(noise_std_value)
            except ValueError:
                pass

        if self.gradientDeviations.text() == '':
            self._problem_data_info.gradient_deviations = None
        else:
            self._problem_data_info.gradient_deviations = self.gradientDeviations.text()

        self._problem_data_info.static_maps = copy(self._static_maps)

    def _load_config(self):
        """Load the settings from the config into the GUI"""
        if self._problem_data_info.noise_std is not None:
            self.noiseStd.setText(str(self._problem_data_info.noise_std))
        if self._problem_data_info.gradient_deviations is not None:
            self.gradientDeviations.setText(str(self._problem_data_info.gradient_deviations))
        self._static_maps = copy(self._problem_data_info.static_maps)
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
        self.staticMaps.clear()
        self.staticMaps.addItems('{}: {}'.format(key, self._static_maps[key]) for key in sorted(self._static_maps))


class AddStaticMapDialog(Ui_AddStaticMapDialog, QDialog):

    def __init__(self, shared_state, parent, static_maps):
        super(AddStaticMapDialog, self).__init__(parent)
        self._shared_state = shared_state
        self._static_maps = static_maps
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

        self._static_maps.update({self.mapNameInput.text(): value})

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

