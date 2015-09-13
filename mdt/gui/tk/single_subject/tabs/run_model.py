from collections import OrderedDict

try:
    #python 2.7
    from Queue import Empty
    from Tkconstants import W, HORIZONTAL, EXTENDED, BOTH, YES
    from Tkinter import StringVar, Listbox, IntVar, Text, Frame, Toplevel, Scrollbar, Pack, Grid, Place
    import ttk
    import tkMessageBox
except ImportError:
    # python 3.4
    from queue import Empty
    from tkinter.constants import W, HORIZONTAL, EXTENDED, BOTH, YES
    from tkinter import StringVar, Listbox, IntVar, Text, Frame, Toplevel, Scrollbar, Pack, Grid, Place
    from tkinter import ttk
    import tkinter.messagebox as tkMessageBox

import copy
from itertools import count
import numbers
import os
import multiprocessing
import mdt
from mdt.gui.tk.utils import SubWindow, TabContainer
from mdt.gui.tk.widgets import FileBrowserWidget, DirectoryBrowserWidget, DropdownWidget, SubWindowWidget, \
    TextboxWidget, YesNonWidget, ListboxWidget, RadioButtonWidget
from mdt.gui.utils import OptimOptions, function_message_decorator
from mdt.utils import MetaOptimizerBuilder, split_image_path
from mot.factory import get_optimizer_by_name

__author__ = 'Robbert Harms'
__date__ = "2015-08-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class RunModelTab(TabContainer):

    def __init__(self, window, cl_process_queue, output_queue):
        super(RunModelTab, self).__init__(window, cl_process_queue, output_queue, 'Run model')

        self._models_ordered_list = mdt.get_models_list()
        self._models = {k: v['description'] for k, v in mdt.get_models_meta_info().items()}
        self._models_default = self._models_ordered_list[0]

        self.optim_options = OptimOptions()
        self._optim_options_window = OptimOptionsWindow(self)

        self._image_vol_chooser = FileBrowserWidget(
            self._tab,
            'image_vol_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select 4d image: ', '(Select the measured diffusion weighted image)')

        self._brain_mask_chooser = FileBrowserWidget(
            self._tab,
            'brain_mask_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select brain mask: ',
            '(To create one see the tab "Generate brain masks")')

        self._protocol_file_chooser = FileBrowserWidget(
            self._tab,
            'protocol_files',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select protocol file: ',
            '(To create one see the tab "Generate protocol file")')

        self._output_dir_chooser = DirectoryBrowserWidget(
            self._tab,
            'output_dir_chooser',
            self._onchange_cb,
            False,
            'Select output folder: ',
            '(Defaults to "output/<mask name>" in\n the same directory as the 4d image)')

        self._model_select_chooser = DropdownWidget(
            self._tab,
            'model_select_chooser',
            self._onchange_cb,
            self._models_ordered_list,
            self._models_default,
            'Select model: ',
            '(Please select a model)')

        self._optim_options_button = SubWindowWidget(
            self._tab,
            'optim_options_button',
            self._optim_options_window,
            'Optimization options: ',
            '(The defaults are loaded from your config file\n these are generally fine)')

        self._io_fields = [self._image_vol_chooser, self._brain_mask_chooser,
                           self._protocol_file_chooser, self._output_dir_chooser]
        self._model_select_fields = [self._model_select_chooser]
        self._required_fields = [self._image_vol_chooser, self._brain_mask_chooser, self._protocol_file_chooser]
        self._run_button = ttk.Button(self._tab, text='Run', command=self._run_model, state='disabled')

    def get_tab(self):
        row_nmr = count()

        label = ttk.Label(self._tab, text="Run model", font=(None, 14))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="Optimize a model to your data.",
                          font=(None, 9, 'italic'))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        for field in self._io_fields:
            field.render(next(row_nmr))
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(8, 3))

        for field in self._model_select_fields:
            field.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(8, 3))
        self._optim_options_button.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(8, 3))
        self._run_button.grid(row=next(row_nmr), sticky='W', pady=(10, 0))

        return self._tab

    def tab_selected(self):
        self._init_path_chooser(self._image_vol_chooser, TabContainer.last_used_dwi_image)
        self._init_path_chooser(self._brain_mask_chooser, TabContainer.last_used_mask)
        self._init_path_chooser(self._brain_mask_chooser, TabContainer.last_used_roi_mask)
        self._init_path_chooser(self._protocol_file_chooser, TabContainer.last_used_protocol)

    def _run_model(self):
        self._run_button.config(state='disabled')
        protocol_ok = self._test_protocol()

        if protocol_ok:
            self._set_last_run_settings()

            model_name = self._model_select_chooser.get_value()

            image_path = self._image_vol_chooser.get_value()
            protocol_path = self._protocol_file_chooser.get_value()
            brain_mask_path = self._brain_mask_chooser.get_value()
            output_dir = self._output_dir_chooser.get_value()

            manager = multiprocessing.Manager()
            finish_queue = manager.Queue()

            run_proc = RunModelProcess(finish_queue, model_name, image_path, protocol_path, brain_mask_path,
                                       output_dir, self.optim_options)
            self._cl_process_queue.put(run_proc)

            def _wait_for_run_completion():
                try:
                    finish_queue.get(block=False)
                    self._run_button.config(state='normal')
                except Empty:
                    self.window.after(100, _wait_for_run_completion)

            self.window.after(100, _wait_for_run_completion)

    def _test_protocol(self):
        model_name = self._model_select_chooser.get_value()
        model = mdt.get_model(model_name)

        protocol = mdt.load_protocol(self._protocol_file_chooser.get_value())
        protocol_sufficient = model.is_protocol_sufficient(protocol)

        if not protocol_sufficient:
            problems = model.get_protocol_problems(protocol)
            tkMessageBox.showerror('Protocol insufficient', "\n".join(['- ' + str(p) for p in problems]))
            return False
        return True

    def _set_last_run_settings(self):
        TabContainer.last_used_dwi_image = self._image_vol_chooser.get_value()
        TabContainer.last_used_protocol = self._protocol_file_chooser.get_value()
        TabContainer.last_used_model_output_folder = self._output_dir_chooser.get_value()
        TabContainer.last_optimized_model = self._model_select_chooser.get_value()

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key

        self._update_global_initial_dir(calling_widget, ['image_vol_chooser', 'brain_mask_chooser', 'protocol_files'])

        if id_key == 'image_vol_chooser':
            path, img_name = split_image_path(calling_widget.get_value())[0:2]

            if not self._brain_mask_chooser.is_valid():
                mask_name = os.path.join(path, img_name + '_mask.nii.gz')
                if os.path.isfile(mask_name):
                    self._brain_mask_chooser.initial_file = mask_name

            if not self._protocol_file_chooser.is_valid():
                prtcl_name = os.path.join(path, img_name + '.prtcl')
                if os.path.isfile(prtcl_name):
                    self._protocol_file_chooser.initial_file = prtcl_name

        if id_key != 'output_dir_chooser':
            if not self._output_dir_chooser.get_value() and self._image_vol_chooser.is_valid() \
                    and self._brain_mask_chooser.is_valid():
                mask_name = os.path.splitext(os.path.basename(self._brain_mask_chooser.get_value()))[0]
                mask_name = mask_name.replace('.nii', '')
                self._output_dir_chooser.initial_dir = os.path.join(os.path.dirname(
                    self._image_vol_chooser.get_value()), 'output', mask_name)

        all_valid = all(field.is_valid() for field in self._required_fields)
        if all_valid:
            self._run_button.config(state='normal')
        else:
            self._run_button.config(state='disabled')


class RunModelProcess(object):

    def __init__(self, finish_queue, model_name, image_path, protocol_path, brain_mask_path, output_dir, optim_options):
        self._finish_queue = finish_queue
        self._model_name = model_name
        self._image_path = image_path
        self._protocol_path = protocol_path
        self._brain_mask_path = brain_mask_path
        self._output_dir = output_dir
        self._optim_options = optim_options

    @function_message_decorator('Starting model fitting, please wait.',
                                'Finished model fitting. You can view the results using the "View results" tab.')
    def __call__(self, *args, **kwargs):
        if self._optim_options.use_model_default_optimizer:
            optimizer = None
        else:
            optimizer = MetaOptimizerBuilder(self._optim_options.get_meta_optimizer_config()).construct()

        mdt.fit_model(self._model_name, self._image_path, self._protocol_path, self._brain_mask_path, self._output_dir,
                      optimizer=optimizer,
                      recalculate=True,
                      only_recalculate_last=self._optim_options.recalculate_all,
                      double_precision=self._optim_options.double_precision)
        self._finish_queue.put('DONE')


class OptimOptionsWindow(SubWindow):

    def __init__(self, parent):
        super(OptimOptionsWindow, self).__init__('Optimization options')
        self._parent = parent
        self._optim_options = copy.copy(parent.optim_options)

    def render(self, window):
        self._optim_options = copy.copy(self._parent.optim_options)

        subframe = ttk.Frame(window)
        subframe.config(padding=(10, 13, 10, 10))
        subframe.grid_columnconfigure(3, weight=1)

        self._optim_routine_chooser = self._get_optim_routine_chooser(subframe)

        self._use_model_default_optimizer = YesNonWidget(
            subframe,
            'use_model_default_optimizer',
            self._onchange_cb,
            'Use model default: ',
            '(Use the default optimizer for the chosen model,\n in cascades, auto-chooses optimizer per submodel)',
            default_val=self._optim_options.use_model_default_optimizer)

        self._patience_box = TextboxWidget(
            subframe,
            'patience_box',
            self._onchange_cb,
            'Patience: ',
            '(The amount of iterations to wait per parameter)',
            default_val=self._optim_options.patience)

        self._extra_optim_runs = TextboxWidget(
            subframe,
            'extra_optim_runs',
            self._onchange_cb,
            'Extra runs: ', '(The additional number of iterations,\n with a smoothing step in between)',
            default_val=self._optim_options.extra_optim_runs)

        default_smoother = None
        for name, val in self._optim_options.smoothing_routines.items():
            if val == self._optim_options.smoother:
                default_smoother = name
                break

        self._smoothing_routine_chooser = DropdownWidget(
            subframe,
            'smoothing_routine_chooser',
            self._onchange_cb,
            self._optim_options.smoothing_routines.keys(),
            default_smoother,
            'Select smoothing routine: ',
            '(Used before each additional optimization iteration)')

        if isinstance(self._optim_options.smoother_size, numbers.Number):
            default_smoother_size = self._optim_options.smoother_size
        else:
            default_smoother_size = ','.join(map(str, self._optim_options.smoother_size))

        self._smoothing_size = TextboxWidget(
            subframe,
            'smoothing_size',
            self._onchange_cb,
            'Smoothing filter size: ', '(Either one integer or three comma separated.\n '
                                       'This is the smoothing filter size in voxels.)',
            default_val=default_smoother_size)

        self._double_precision = RadioButtonWidget(
            subframe,
            'double_precision',
            self._onchange_cb,
            'Float precision: ',
            '(Specify the precision of the calculations)',
            OrderedDict([('Float', False), ('Double', True)]),
            default_val=self._optim_options.double_precision)

        self._recalculate_all = YesNonWidget(
            subframe,
            'recalculate_all',
            self._onchange_cb,
            'Recalculate all: ',
            '(For Cascades, if yes, recalculate all maps,\n if no, only the last)',
            default_val=self._optim_options.recalculate_all)

        self._devices_chooser = ListboxWidget(
            subframe,
            'cl_environment_chooser',
            self._onchange_cb,
            (),
            '',
            EXTENDED,
            3,
            'Select OpenCL devices: ',
            '(Select the devices you would like to use)')

        default_cl_environments = []
        cl_environment_names = list(self._optim_options.cl_environments.keys())
        for ind in self._optim_options.cl_envs_indices:
            default_cl_environments.append(cl_environment_names[ind])

        self._devices_chooser.set_items(list(self._optim_options.cl_environments.keys()),
                                        default_items=default_cl_environments)

        self._optimizer_fields = [self._optim_routine_chooser, self._patience_box, self._extra_optim_runs,
                            self._smoothing_routine_chooser, self._smoothing_size]
        extra_fields = [self._double_precision, self._recalculate_all, self._devices_chooser]

        button_frame = self._get_button_frame(subframe, window)

        row_nmr = count()
        label = ttk.Label(subframe, text="Optimization options", font=(None, 14))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        label = ttk.Label(subframe, text="Options for the optimization routines, these are advanced settings.",
                          font=(None, 9, 'italic'))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        self._use_model_default_optimizer.render(next(row_nmr))
        for field in self._optimizer_fields:
            field.render(next(row_nmr))
            field.set_state('disabled' if self._optim_options.use_model_default_optimizer else 'normal')

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        for field in extra_fields:
            field.render(next(row_nmr))

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        button_frame.grid(row=next(row_nmr), sticky='W', pady=(10, 0), columnspan=4)

        subframe.pack(fill=BOTH, expand=YES)

    def _get_optim_routine_chooser(self, parent):
        default_optimizer = None
        for name, val in self._optim_options.optim_routines.items():
            if val == self._optim_options.optimizer:
                default_optimizer = name
                break

        return DropdownWidget(
            parent,
            'optim_routine_chooser',
            self._onchange_cb,
            self._optim_options.optim_routines.keys(),
            default_optimizer,
            'Optimization routine: ',
            '(Select the routine you would like to use)')

    def _get_button_frame(self, parent, window):
        def accept():
            self._parent.optim_options = self._optim_options
            window.destroy()

        button_frame = ttk.Frame(parent)
        ok_button = ttk.Button(button_frame, text='Accept', command=accept, state='normal')
        cancel_button = ttk.Button(button_frame, text='Cancel', command=window.destroy, state='normal')
        ok_button.grid(row=0)
        cancel_button.grid(row=0, column=1, padx=(10, 0))

        return button_frame

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key

        if id_key == 'optim_routine_chooser':
            optimizer = self._optim_options.optim_routines[self._optim_routine_chooser.get_value()]
            self._optim_options.optimizer = optimizer

            optimizer_class = get_optimizer_by_name(optimizer)
            self._patience_box.set_value(optimizer_class.default_patience)

        elif id_key == 'use_model_default_optimizer':
            self._optim_options.use_model_default_optimizer = calling_widget.get_value()
            for field in self._optimizer_fields:
                field.set_state('disabled' if self._optim_options.use_model_default_optimizer else 'normal')

        elif id_key == 'patience_box':
            try:
                self._optim_options.patience = int(calling_widget.get_value())
            except ValueError:
                pass

        elif id_key == 'recalculate_all':
            self._optim_options.recalculate_all = calling_widget.get_value()

        elif id_key == 'extra_optim_runs':
            try:
                self._optim_options.extra_optim_runs = int(calling_widget.get_value())
            except ValueError:
                pass

        elif id_key == 'smoothing_routine_chooser':
            smoother = self._optim_options.smoothing_routines[calling_widget.get_value()]
            self._optim_options.smoother = smoother
            self._smoothing_size.set_value(1)

        elif id_key == 'smoothing_size':
            try:
                self._optim_options.smoother_size = int(calling_widget.get_value())
            except ValueError:
                try:
                    self._optim_options.smoother_size = [int(v) for v in calling_widget.get_value().split(',')][0:3]
                except ValueError:
                    pass

        elif id_key == 'double_precision':
            self._optim_options.double_precision = calling_widget.get_value()

        elif id_key == 'cl_environment_chooser':
            chosen_keys = self._devices_chooser.get_value()
            l = [ind for ind, key in enumerate(self._optim_options.cl_environments.keys()) if key in chosen_keys]
            if l:
                self._optim_options.cl_envs_indices = l