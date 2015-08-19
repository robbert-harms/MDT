from ScrolledText import ScrolledText
from Tkconstants import W, E, BOTH, FALSE, YES, HORIZONTAL, VERTICAL, END, EXTENDED, BROWSE, INSERT
from Tkinter import Tk, StringVar, IntVar, Listbox, TclError
import Tkinter
import copy
import glob
import logging
import os
import threading
import tkFileDialog
import tkMessageBox
import ttk
import sys
import multiprocessing
import mdt
import mdt.utils
from math import log10
import platform
from numpy import genfromtxt
import nibabel as nib
import numpy as np
from mdt import load_protocol, load_dwi, load_brain_mask, create_median_otsu_brain_mask, \
    create_slice_roi, load_protocol_bval_bvec, view_results_slice, fit_model, concatenate_mri_sets
import mot.cl_environments
from mot.cl_routines.filters.median import MedianFilter
from mot.load_balance_strategies import EvenDistribution
from mot.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from mot.cl_routines.optimizing.meta_optimizer import MetaOptimizer
from mot.cl_routines.optimizing.nmsimplex import NMSimplex
from mot.cl_routines.optimizing.powell import Powell
from mot.cl_routines.filters.mean import MeanFilter
from mdt.visualization import MapsVisualizer
import mdt.protocols
import mdt.configuration


__author__ = 'Robbert Harms'
__date__ = "2014-10-01"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


mdt.utils.setup_logging(disable_existing_loggers=True)


def get_window():
    window = Tk()
    s = ttk.Style()
    try:
        s.theme_use('clam')
    except TclError:
        pass

    window.wm_title("Diffusion MRI Toolbox")
    notebook = ttk.Notebook(window)
    window.resizable(width=FALSE, height=FALSE)
    window.update_idletasks()
    width = 900
    height = 600
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    tabs = (RunModelTab(window),
            GenerateBrainMaskTab(window),
            GenerateROIMaskTab(window),
            GenerateProtocolFileTab(window),
            ConcatenateShellsTab(window),
            ViewResultsTab(window))

    for tab in tabs:
        notebook.add(tab.get_tab(), text=tab.tab_name)

    notebook.pack(fill=BOTH, expand=YES)

    txt = ScrolledText(window, height=15)
    txt.pack(fill=BOTH, expand=YES)

    real_stdout = sys.stdout

    def start_stdout_redirect():
        sys.stdout = StdoutTextWrapper(txt)

        for module in ('mdt', 'mot'):
            logger = logging.getLogger(module)
            console = logging.StreamHandler(stream=sys.stdout)
            logger.addHandler(console)

        from mdt import VERSION
        welcome_str = 'Welcome to MDT version ' + VERSION
        print(welcome_str)
        print('-' * len(welcome_str))
        print('')
        print('This area is reserved for print and logging output from MDT.')


    window.after(100, start_stdout_redirect)

    def on_closing():
        sys.stdout = real_stdout
        window.destroy()
    window.protocol("WM_DELETE_WINDOW", on_closing)

    return window


class StdoutTextWrapper(object):

    def __init__(self, text_widget):
        self._text_box = text_widget

    def write(self, string):
        self._text_box.configure(state='normal')
        self._text_box.insert(END, string)
        self._text_box.configure(state='disabled')
        self._text_box.see(END)

    def flush(self):
        pass


class TabContainer(object):

    def __init__(self, window, tab_name):
        self.window = window
        self.tab_name = tab_name
        self._tab = ttk.Frame()
        self._tab.config(padding=(10, 13, 10, 10))
        self._tab.grid_columnconfigure(3, weight=1)

    def get_tab(self):
        """Get the tab frame for this tab object"""
        pass


class RunModelTab(TabContainer):

    def __init__(self, window):
        super(RunModelTab, self).__init__(window, 'Run model')

        self._models_ordered_list = mdt.get_models_list()
        self._models = {k: v['description'] for k, v in mdt.get_models_meta_info().items()}
        self._models_default = self._models_ordered_list[0]

        self._optim_routine_default = '-- Auto --'
        self._optim_routines = {'-- Auto --': NMSimplex,
                                'Nelder-Mead Simplex': NMSimplex,
                                'Levenberg Marquardt': LevenbergMarquardt,
                                'Powell\'s method': Powell}

        self._smoothing_routines = {'Median filter': MedianFilter,
                                    'Mean filter': MeanFilter}

        self._cl_environments = {}
        self._cl_environments_keys = []
        cl_environments = mot.cl_environments.CLEnvironmentFactory.all_devices()
        i = 1
        for env in cl_environments:
            s = repr(i) + ') '
            s += 'GPU' if env.is_gpu else 'CPU'
            s += ' - ' + env.device.name + ' (' + env.platform.name + ')'
            self._cl_environments.update({s: env})
            self._cl_environments_keys.append(s)
            i += 1

        self._optim_options = OptimOptions()
        self._optim_options.optim_routine_index = self._optim_routine_default
        self._optim_options.smoother_index = 'Median filter'
        self._optim_options.cl_envs_indices = [self._get_gpu_device_key()]
        self._optim_options_window = OptimOptionsWindow(self)

        self._image_vol_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_cb('image_vol_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select 4d image: ', '(Select the measured MRI volume)')

        self._brain_mask_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_cb('brain_mask_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select brain mask: ',
            '(If you don\'t have one, generate one\n on the tab "Generate brain masks")')

        self._protocol_file_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3:
            self._onchange_cb('protocol_file_chooser'),
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select protocol file: ',
            '(If you don\'t have one, generate one\n on the tab "Generate protocol file")')

        self._output_dir_chooser = DirectoryBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3:
            self._onchange_cb('output_dir_chooser'),
            False,
            'Select output folder: ',
            '(Defaults to "output/<mask name>" in\n the same directory as the 4d image)')

        self._model_select_chooser = DropdownWidget(
            self._tab,
            lambda val: self._onchange_cb('model_select_chooser'),
            self._models_ordered_list,
            self._models_default,
            'Select model: ',
            '(Please select a model)')

        self._optim_options_button = SubWindowWidget(
            self._tab,
            self._optim_options_window,
            'Optimization options: ',
            '(The defaults are generally fine)')

        self._io_fields = [self._image_vol_chooser, self._brain_mask_chooser,
                           self._protocol_file_chooser, self._output_dir_chooser]
        self._model_select_fields = [self._model_select_chooser]

        self._required_fields = [self._image_vol_chooser, self._brain_mask_chooser, self._protocol_file_chooser]

        self._run_button = ttk.Button(self._tab, text='Run', command=self._run_model, state='disabled')

    def get_tab(self):
        row = 0
        label = ttk.Label(self._tab, text="Run model", font=(None, 14))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1

        for field in self._io_fields:
            field.render(row)
            row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(8, 3))
        row += 1

        for field in self._model_select_fields:
            field.render(row)
            row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(8, 3))
        row += 1

        self._optim_options_button.render(row)
        row += 1

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(8, 3))
        row += 1

        self._run_button.grid(row=row, sticky='W', pady=(10, 0))

        return self._tab

    def _run_model(self):
        self._run_button.config(state='disabled')

        problem_data = mdt.load_problem_data(mdt.load_dwi(self._image_vol_chooser.get_value()),
                                             mdt.load_protocol(self._protocol_file_chooser.get_value()),
                                             mdt.load_brain_mask(self._brain_mask_chooser.get_value()),)
        output_folder = self._output_dir_chooser.get_value()

        recalculate = True
        only_recalculate_last = self._optim_options.recalculate_all
        optimizer = self._get_optimizer()

        model_key = self._model_select_chooser.get_value()
        model = mdt.get_model(model_key)

        protocol_sufficient = model.is_protocol_sufficient(problem_data.protocol)
        if protocol_sufficient:
            fit_model(model, problem_data, output_folder, optimizer=optimizer, recalculate=recalculate,
                only_recalculate_last=only_recalculate_last)
            tkMessageBox.showinfo('Calculation completed', 'The calculations have been completed.')
        else:
            problems = model.get_protocol_problems(problem_data.protocol)
            tkMessageBox.showerror('Protocol insufficient', "\n".join(['- ' + str(p) for p in problems]))
        self._run_button.config(state='normal')

    def _get_optimizer(self):
        optimizer = MetaOptimizer()

        sub_opt = self._optim_routines[self._optim_options.optim_routine_index]
        sub_opt = sub_opt(patience=self._optim_options.patience)
        optimizer.optimizer = sub_opt

        smoother = self._smoothing_routines[self._optim_options.smoother_index]
        optimizer.smoother = smoother([self._optim_options.smoother_size] * 3)

        optimizer.extra_optim_runs = self._optim_options.extra_optim_runs
        optimizer.load_balancer = EvenDistribution()

        envs = []
        for key in self._optim_options.cl_envs_indices:
            envs.append(self._cl_environments[key])
        optimizer.cl_environments = envs

        return optimizer

    def _get_gpu_device_key(self):
        for key, env in self._cl_environments.items():
            if env.is_gpu:
                return key
        return self._cl_environments_keys[0]

    def _onchange_cb(self, widget_name):
        if widget_name is not 'output_dir_chooser':
            if not self._output_dir_chooser.get_value() and self._image_vol_chooser.is_valid() \
                    and self._brain_mask_chooser.is_valid():
                mask_name = os.path.splitext(os.path.basename(self._brain_mask_chooser.get_value()))[0]
                mask_name = mask_name.replace('.nii', '')
                self._output_dir_chooser.initial_dir = os.path.join(os.path.dirname(
                    self._image_vol_chooser.get_value()), 'output', mask_name)

        all_valid = True
        for f in self._required_fields:
            if not f.is_valid():
                all_valid = False
                break

        if all_valid:
            self._run_button.config(state='normal')
        else:
            self._run_button.config(state='disabled')


class SubWindow(object):

    def __init__(self, window_title):
        self._window_title = window_title

    @property
    def window_title(self):
        return self._window_title

    def render(self, toplevel):
        """Render the items in the given frame"""


class OptimOptions(object):

    def __init__(self):
        self.patience = 125
        self.recalculate_all = True
        self.optim_routine_index = None
        self.extra_optim_runs = 0
        self.smoother_index = None
        self.smoother_size = 1
        self.cl_envs_indices = []


class OptimOptionsWindow(SubWindow):

    def __init__(self, parent):
        super(OptimOptionsWindow, self).__init__('Optimization options')
        self._parent = parent
        self._optim_options = copy.copy(parent._optim_options)
        self._optim_routine_chooser = None
        self._patience_box = None
        self._recalculate_all = None

    def render(self, toplevel):
        self._optim_options = copy.copy(self._parent._optim_options)

        frame = ttk.Frame(toplevel)
        frame.config(padding=(10, 13, 10, 10))
        frame.grid_columnconfigure(3, weight=1)

        self._optim_routine_chooser = DropdownWidget(
            frame,
            lambda val: self._onchange_cb('optim_routine_chooser'),
            self._parent._optim_routines.keys(),
            self._optim_options.optim_routine_index,
            'Select optimization routine: ',
            '(Auto selects the best optimizer for the chosen model)')

        self._patience_box = TextboxWidget(
            frame,
            lambda trc1, trc2, trc3: self._onchange_cb('patience_box'),
            'Patience: ', '(The amount of iterations to wait per parameter)',
            default_val=self._optim_options.patience)

        self._recalculate_all = YesNonWidget(
            frame,
            lambda: self._onchange_cb('recalculate_all'),
            'Recalculate all: ',
            '(If yes, recalculate all maps in the cascade, if no, only the last)',
            default_val=self._optim_options.recalculate_all)

        self._extra_optim_runs = TextboxWidget(
            frame,
            lambda trc1, trc2, trc3: self._onchange_cb('extra_optim_runs'),
            'Extra runs: ', '(The additional number of iterations,\n with a smoothing step in between)',
            default_val=self._optim_options.extra_optim_runs)

        self._smoothing_routine_chooser = DropdownWidget(
            frame,
            lambda val: self._onchange_cb('smoothing_routine_chooser'),
            self._parent._smoothing_routines.keys(),
            self._optim_options.smoother_index,
            'Select smoothing routine: ',
            '(Used before each additional optimization iteration)')

        self._smoothing_size = TextboxWidget(
            frame,
            lambda trc1, trc2, trc3: self._onchange_cb('smoothing_size'),
            'smoothing filter size: ', '(The size in voxels in all dimensions for the smoothing filter)',
            default_val=self._optim_options.smoother_size)

        self._devices_chooser = ListboxWidget(
            frame,
            lambda val: self._onchange_cb('cl_environment_chooser'),
            (),
            '',
            EXTENDED,
            3,
            'Select OpenCL devices: ',
            '(Select the devices you would like to use)')
        self._devices_chooser.set_items(self._parent._cl_environments_keys,
                                        default_items=self._optim_options.cl_envs_indices)

        fields = [self._optim_routine_chooser, self._patience_box, self._recalculate_all, self._extra_optim_runs,
                  self._smoothing_routine_chooser, self._smoothing_size, self._devices_chooser]

        def accept():
            self._accept()
            toplevel.destroy()

        button_frame = ttk.Frame(frame)
        ok_button = ttk.Button(button_frame, text='Accept', command=accept, state='normal')
        cancel_button = ttk.Button(button_frame, text='Cancel', command=toplevel.destroy, state='normal')
        ok_button.grid(row=0)
        cancel_button.grid(row=0, column=1, padx=(10, 0))

        row = 0
        label = ttk.Label(frame, text="Optimization options", font=(None, 14))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1
        ttk.Separator(frame, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1
        for field in fields:
            field.render(row)
            row += 1
        ttk.Separator(frame, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1
        button_frame.grid(row=row, sticky='W', pady=(10, 0), columnspan=4)
        frame.pack(fill=BOTH, expand=YES)

    def _onchange_cb(self, widget_name):
        if widget_name == 'optim_routine_chooser':
            optim_routine = self._optim_routine_chooser.get_value()
            if optim_routine != self._parent._optim_routine_default:
                sub_opt = self._parent._optim_routines[self._optim_routine_chooser.get_value()]
                self._patience_box.set_value(sub_opt.patience)
            else:
                self._patience_box.set_value(125)

        self._optim_options.patience = self._patience_box.get_value()
        self._optim_options.recalculate_all = self._recalculate_all.get_value()
        self._optim_options.optim_routine_index = self._optim_routine_chooser.get_value()
        self._optim_options.extra_optim_runs = self._extra_optim_runs.get_value()
        self._optim_options.smoother_index = self._smoothing_routine_chooser.get_value()
        self._optim_options.smoother_size = self._smoothing_size.get_value()
        self._optim_options.cl_envs_indices = self._devices_chooser.get_value()

        if self._optim_options.patience:
            try:
                self._optim_options.patience = int(self._optim_options.patience)
            except ValueError:
                self._optim_options.patience = 1
        if self._optim_options.extra_optim_runs:
            try:
                self._optim_options.extra_optim_runs = int(self._optim_options.extra_optim_runs)
            except ValueError:
                self._optim_options.extra_optim_runs = 1
        if self._optim_options.smoother_size:
            try:
                self._optim_options.smoother_size = int(self._optim_options.smoother_size)
            except ValueError:
                self._optim_options.smoother_size = 1

        if self._optim_options.patience < 0:
            self._optim_options.patience = 1
        if self._optim_options.extra_optim_runs < 0:
            self._optim_options.extra_optim_runs = 0
        if self._optim_options.smoother_size < 1:
            self._optim_options.smoother_size = 1
        if not self._optim_options.cl_envs_indices:
            self._optim_options.cl_envs_indices = self._parent._cl_environments_keys

    def _accept(self):
        self._parent._optim_options = self._optim_options


class GenerateBrainMaskTab(TabContainer):

    def __init__(self, window):
        super(GenerateBrainMaskTab, self).__init__(window, 'Generate brain mask')

        self._dimensions = {}
        self._dimension_shape = []

        self._image_vol_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_bm_cb('image_vol_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select 4d image: ', '(Select the measured 4d diffusion weighted image)')

        self._protocol_file_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3:
            self._onchange_bm_cb('protocol_file_chooser'),
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select protocol file: ', '(Please see the tab "Generate protocol file")')

        self._output_bm_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_bm_cb('output_bm_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select output file: ', '(Default is <volume_name>_mask.nii.gz)', dialog_type=FileBrowserWidget.SAVE)

        self._median_radius_box = TextboxWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_bm_cb('median_radius_box'),
            'Median radius: ', '(Radius (in voxels) of the applied median filter)',
            default_val='4')

        self._numpasses_box = TextboxWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_bm_cb('median_radius_box'),
            'Number of passes: ', '(Number of median filter passes)',
            default_val='4')

        self._threshold_box = TextboxWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_bm_cb('threshold'),
            'Final threshold: ', '(Additional masking threshold)',
            default_val='0')

        self._mask_items = [self._image_vol_chooser, self._protocol_file_chooser, self._output_bm_chooser,
                            self._median_radius_box, self._numpasses_box, self._threshold_box]

        self._whole_mask_buttons_frame = ttk.Frame(self._tab)
        self._run_whole_brain_mask_button = ttk.Button(self._whole_mask_buttons_frame, text='Generate',
                                                       command=self._run_whole_brain_mask, state='disabled')
        self._view_brain_mask_button = ttk.Button(self._whole_mask_buttons_frame, text='View mask',
                                                  command=self._view_brain_mask, state='disabled')
        self._run_whole_brain_mask_button.grid(row=0)
        self._view_brain_mask_button.grid(row=0, column=1, padx=(10, 0))

    def get_tab(self):
        row = 0
        label = ttk.Label(self._tab, text="Generate brain mask", font=(None, 14))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1

        label = ttk.Label(self._tab, text="Creates a mask for a whole brain using the median-otsu algorithm.",
                          font=(None, 9, 'italic'))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1
        for field in self._mask_items:
            field.render(row)
            row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1
        self._whole_mask_buttons_frame.grid(row=row, sticky='W', pady=(10, 0), columnspan=4)
        return self._tab

    def _onchange_bm_cb(self, cb_name):
        if cb_name == 'image_vol_chooser':
            img_name = self._image_vol_chooser.get_value()
            img_name = os.path.splitext(img_name)[0]
            img_name = img_name.replace('.nii', '')
            img_name += '_mask.nii.gz'
            self._output_bm_chooser.initial_file = img_name

        if os.path.isfile(self._output_bm_chooser.get_value())\
            and self._image_vol_chooser.is_valid()\
            and os.path.isfile(self._protocol_file_chooser.get_value()):
            self._view_brain_mask_button.config(state='normal')
        else:
            self._view_brain_mask_button.config(state='disabled')

        if self._image_vol_chooser.is_valid() and self._output_bm_chooser.is_valid():
            self._run_whole_brain_mask_button.config(state='normal')
            self._view_brain_mask_button.config(state='normal')
        else:
            self._run_whole_brain_mask_button.config(state='disabled')

    def _run_whole_brain_mask(self):
        self._run_whole_brain_mask_button.config(state='disabled')
        volume = self._image_vol_chooser.get_value()
        prtcl = self._protocol_file_chooser.get_value()
        output_fname = self._output_bm_chooser.get_value()

        filter_mean_radius = 4
        if self._median_radius_box.is_valid():
            filter_mean_radius = int(self._median_radius_box.get_value())
            filter_mean_radius = max(filter_mean_radius, 1)

        filter_passes = 4
        if self._numpasses_box.is_valid():
            filter_passes = int(self._numpasses_box.get_value())
            filter_passes = max(filter_passes, 1)

        threshold = 0
        if self._threshold_box.is_valid():
            threshold = int(self._threshold_box.get_value())
            threshold = max(threshold, 0)

        def create_brain_mask():
            print('---------------------')
            create_median_otsu_brain_mask(volume, prtcl, output_fname, median_radius=filter_mean_radius,
                                          numpass=filter_passes, mask_threshold=threshold)
            print('---------------------')

        mask_create_thread = threading.Thread(target=create_brain_mask)
        mask_create_thread.start()

        self._run_whole_brain_mask_button.config(state='normal')

    def _view_brain_mask(self):
        mask = self._output_bm_chooser.get_value()
        if os.path.isfile(mask):
            image_data = load_dwi(self._image_vol_chooser.get_value())[0]
            mask = np.expand_dims(load_brain_mask(self._output_bm_chooser.get_value()), axis=3)
            masked_image = image_data * mask
            s = image_data.shape
            MapsVisualizer({'Masked': masked_image,
                            'DWI': image_data}).show(dimension=1, slice_ind=s[1]/2)


class GenerateROIMaskTab(TabContainer):

    def __init__(self, window):
        super(GenerateROIMaskTab, self).__init__(window, 'Generate ROI mask')

        self._dimensions = {}
        self._dimension_shape = []

        self._brain_mask_vol_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_roi_cb('brain_mask_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select brain mask: ', '(Please see tab "Generate brain mask")')

        self._roi_dimension_index = DropdownWidget(
            self._tab,
            lambda val: self._onchange_roi_cb('roi_dimension_chooser'),
            (),
            '',
            'Select dimension: ',
            '(Select a brain mask first)')

        self._roi_slice_index = ListboxWidget(
            self._tab,
            lambda val: self._onchange_roi_cb('roi_slice_chooser'),
            (),
            '',
            BROWSE,
            1,
            'Select slice: ',
            '(Select a brain mask first)')

        self._output_roi_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_roi_cb('roi_output_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select output file: ', '(Default is <mask_name>_<dim>_<slice>.nii.gz)',
            dialog_type=FileBrowserWidget.SAVE)

        self._roi_items = [self._brain_mask_vol_chooser, self._roi_dimension_index,
                           self._roi_slice_index, self._output_roi_chooser, ]

        self._buttons_frame = ttk.Frame(self._tab)
        self._run_slice_roi_button = ttk.Button(self._buttons_frame, text='Generate',
                                                command=self._run_slice_roi, state='disabled')
        self._run_slice_roi_button.grid(row=0)

    def get_tab(self):
        row = 0
        label = ttk.Label(self._tab, text="Generate ROI mask", font=(None, 14))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1
        label = ttk.Label(self._tab, text="Create a brain mask with a Region Of Interest "
                                          "that only includes the voxels in the selected slice. ",
                          font=(None, 9, 'italic'))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1
        for field in self._roi_items:
            field.render(row)
            row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(10, 3))
        row += 1
        self._buttons_frame.grid(row=row, sticky='W', pady=(10, 0), columnspan=4)
        return self._tab

    def _onchange_roi_cb(self, cb_name):
        brain_mask_fname = self._brain_mask_vol_chooser.get_value()

        if brain_mask_fname:
            if cb_name == 'brain_mask_chooser':
                mask = nib.load(brain_mask_fname)
                shape = mask.shape
                self._dimension_shape = shape
                self._dimensions = {}
                items = []
                for i in range(len(shape)):
                    items.append('{} ({} items)'.format(i, shape[i]))
                    self._dimensions.update({'{} ({} items)'.format(i, shape[i]): i})
                self._roi_dimension_index.set_items(items, items[0])
                self._update_roi_slice_chooser()

            elif cb_name == 'roi_dimension_chooser':
                self._update_roi_slice_chooser()

            brain_mask = self._brain_mask_vol_chooser.get_value()
            roi_dim = int(self._dimensions[self._roi_dimension_index.get_value()])
            roi_slice = int(self._roi_slice_index.get_value()[0])
            mask_name = os.path.splitext(os.path.basename(brain_mask))[0].replace('.nii', '')
            output_fname = os.path.join(os.path.dirname(brain_mask), mask_name + '_' + repr(roi_dim) + '_' +
                                        repr(roi_slice)) + '.nii.gz'
            self._output_roi_chooser.initial_file = output_fname

        all_valid = True
        for field in self._roi_items:
            if not field.is_valid():
                all_valid = False
                break

        if all_valid:
            self._run_slice_roi_button.config(state='normal')
        else:
            self._run_slice_roi_button.config(state='disabled')

    def _update_roi_slice_chooser(self):
        dimension = self._dimensions[self._roi_dimension_index.get_value()]
        length = self._dimension_shape[dimension]
        self._roi_slice_index.set_items(range(length), int(length/2))

    def _run_slice_roi(self):
        self._run_slice_roi_button.config(state='disabled')
        brain_mask = self._brain_mask_vol_chooser.get_value()
        roi_dim = int(self._dimensions[self._roi_dimension_index.get_value()])
        roi_slice = int(self._roi_slice_index.get_value()[0])
        output_fname = self._output_roi_chooser.get_value()
        create_slice_roi(brain_mask, roi_dim, roi_slice, output_fname, overwrite_if_exists=True)
        tkMessageBox.showinfo('Action completed', 'The action has been completed.')
        self._run_slice_roi_button.config(state='normal')


class GenerateProtocolFileTab(TabContainer):

    def __init__(self, window):
        super(GenerateProtocolFileTab, self).__init__(window, 'Generate protocol file')

        self._bvec_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_to_protocol('bvec_chooser'),
            FileBrowserWidget.common_file_types('bvec'),
            'Select bvec file: ', '(Select the gradient vectors file)',
            dialog_type=FileBrowserWidget.SAVE)

        self._bval_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_to_protocol('bval_chooser'),
            FileBrowserWidget.common_file_types('bval'),
            'Select bval file: ', '(Select the gradient b-values file)',
            dialog_type=FileBrowserWidget.SAVE)

        self._bval_scale_box = TextboxWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_to_protocol('bval_scale_box'),
            'B-value scale factor: ', '(We expect the the b-values in the\nresult protocol in s/m^2)',
            default_val='1e6')

        self._output_protocol_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_to_protocol('output_protocol_chooser'),
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select output (protocol) file: ', '(Default is <bvec_name>.prtcl)', dialog_type=FileBrowserWidget.SAVE)

        self._estimate_timings = YesNonWidget(self._tab,
                                              lambda: self._onchange_to_protocol('estimate_sequence_timing'),
                                              'Add sequence timings: ',
                                              '(Guess the sequence timings\n (G, Delta, delta) from the b values)')

        self._maxG_box = TextboxWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_to_protocol('Delta_box'),
            'Max G: ', '(Specify the maximum gradient\n amplitude (mT/m))',
            default_val='40', state='disabled')

        self._Delta_box = TextboxWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_to_protocol('Delta_box'),
            'Big Delta: ', '(Optionally, use this Delta\n for the sequence timings (ms))',
            default_val='', state='disabled')

        self._delta_box = TextboxWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_to_protocol('delta_box'),
            'Small delta: ', '(Optionally, use this delta\n for the sequence timings (ms))',
            default_val='', state='disabled')

        self._te_box = TextboxWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_to_protocol('te_box'),
            'TE: ', '(Optionally, use this TE\n for the echo time (ms))',
            default_val='', state='disabled')

        self._to_protocol_items = (self._bvec_chooser, self._bval_chooser, self._bval_scale_box,
                                   self._output_protocol_chooser, self._estimate_timings, self._maxG_box,
                                   self._Delta_box, self._delta_box, self._te_box)

        self._buttons_frame = ttk.Frame(self._tab)
        self._generate_prtcl_button = ttk.Button(self._buttons_frame, text='Generate protocol',
                                                 command=self._generate_protocol, state='disabled')

        self._generate_bval_bvec_button = ttk.Button(self._buttons_frame, text='Generate bval/bvec',
                                                     command=self._generate_bval_bvec, state='disabled')

        self._view_results_button = ttk.Button(self._buttons_frame, text='View protocol file',
                                               command=self._view_results, state='disabled')
        self._generate_prtcl_button.grid(row=0)
        self._view_results_button.grid(row=0, column=1, padx=(10, 0))
        self._generate_bval_bvec_button.grid(row=0, column=2, padx=(10, 0))

    def get_tab(self):
        row = 0
        label = ttk.Label(self._tab, text="Generate protocol file", font=(None, 14))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1
        for field in self._to_protocol_items:
            field.render(row)
            row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(10, 3))
        row += 1
        self._buttons_frame.grid(row=row, sticky='W', columnspan=4, pady=(10, 0))
        return self._tab

    def _onchange_to_protocol(self, cb_name):
        if cb_name == 'bvec_chooser':
            self._output_protocol_chooser.initial_file = os.path.splitext(self._bvec_chooser.get_value())[0] + '.prtcl'

        if cb_name == 'bval_chooser':
            bval_file = self._bval_chooser.get_value()
            if os.path.isfile(bval_file):
                data = genfromtxt(bval_file)
                if log10(np.mean(data)) >= 8:
                    self._bval_scale_box.set_value('1')
                else:
                    self._bval_scale_box.set_value('1e6')
            else:
                self._bval_scale_box.set_value('1e-6')

        if cb_name == 'estimate_sequence_timing':
            estimate_st = self._estimate_timings.get_value()
            if estimate_st:
                self._maxG_box.set_state('normal')
                self._Delta_box.set_state('normal')
                self._delta_box.set_state('normal')
                self._te_box.set_state('normal')
            else:
                self._maxG_box.set_state('disabled')
                self._Delta_box.set_state('disabled')
                self._delta_box.set_state('disabled')
                self._te_box.set_state('disabled')

        if self._output_protocol_chooser.is_valid():
            fname = self._output_protocol_chooser.get_value()
            if os.path.isfile(fname):
                self._view_results_button.config(state='normal')
            else:
                self._view_results_button.config(state='disabled')

        all_valid = True
        for field in self._to_protocol_items:
            if not field.is_valid():
                all_valid = False
                break

        if all_valid:
            self._generate_prtcl_button.config(state='normal')
            self._generate_bval_bvec_button.config(state='normal')
        else:
            self._generate_prtcl_button.config(state='disabled')
            self._generate_bval_bvec_button.config(state='disabled')

    def _generate_protocol(self):
        self._generate_prtcl_button.config(state='disabled')
        bvec_fname = self._bvec_chooser.get_value()
        bval_fname = self._bval_chooser.get_value()
        output_fname = self._output_protocol_chooser.get_value()
        estimate_st = self._estimate_timings.get_value()
        bval_scale = float(self._bval_scale_box.get_value())

        protocol = load_protocol_bval_bvec(bvec=bvec_fname, bval=bval_fname, bval_scale=bval_scale)
        if estimate_st:
            maxG = float(self._maxG_box.get_value()) / 1000.0

            if self._te_box.get_value():
                protocol.add_column('TE', float(self._te_box.get_value()) / 1000.0)
            if self._Delta_box.get_value():
                protocol.add_column('Delta', float(self._Delta_box.get_value()) / 1000.0)
            if self._delta_box.get_value():
                protocol.add_column('delta', float(self._delta_box.get_value()) / 1000.0)

            protocol.add_estimated_protocol_params(maxG=maxG)

        mdt.protocols.write_protocol(protocol, output_fname)

        tkMessageBox.showinfo('Action completed', 'The protocol file has been written.')
        self._view_results_button.config(state='normal')
        self._generate_prtcl_button.config(state='normal')

    def _generate_bval_bvec(self):
        self._generate_bval_bvec_button.config(state='disabled')
        bvec_fname = self._bvec_chooser.get_value()
        bval_fname = self._bval_chooser.get_value()
        prtcl_fname = self._output_protocol_chooser.get_value()

        protocol = load_protocol(prtcl_fname)
        bval_scale = float(self._bval_scale_box.get_value())

        mdt.protocols.write_bvec_bval(protocol, bvec_fname, bval_fname, bval_scale=bval_scale)
        tkMessageBox.showinfo('Action completed', 'The bval and bvec files have been written.')
        self._generate_bval_bvec_button.config(state='normal')

    def _view_results(self):
        output_fname = self._output_protocol_chooser.get_value()
        t = Tkinter.Toplevel(self.window)
        w = Tkinter.Text(t)
        if os.path.isfile(output_fname):
            w.insert(INSERT, self._format_columns(open(output_fname, 'r').read()))
        w.pack(side='top', fill='both', expand=True)
        t.wm_title('Generated protocol (read-only)')

    def _format_columns(self, table):
        return table.replace("\t", "\t\t\t\t")


class ConcatenateShellsTab(TabContainer):

    def __init__(self, window):
        super(ConcatenateShellsTab, self).__init__(window, 'Concatenate shells')

        self._image_1_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_concat('image_1_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select 4d image 1: ', '(Select the first measured 4d image)')

        self._protocol_1_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_concat('protocol_1_chooser'),
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select protocol 1: ', '(Select the first protocol file)')

        self._image_2_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_concat('image_2_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select 4d image 2: ', '(Select the second measured 4d image)')

        self._protocol_2_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_concat('protocol_2_chooser'),
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select protocol 2: ', '(Select the second protocol file)')

        self._output_image_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_concat('output_image_chooser'),
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select output volume: ', '(Default is <vol_1><vol_2>.nii.gz)', dialog_type=FileBrowserWidget.SAVE)

        self._output_protocol_chooser = FileBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3: self._onchange_concat('output_protocol_chooser'),
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select output protocol: ', '(Default is <prtcl_1>_<prtcl_2>.prtcl)', dialog_type=FileBrowserWidget.SAVE)

        self._validate_fields = [self._image_1_chooser, self._protocol_1_chooser,
                                 self._image_2_chooser, self._protocol_2_chooser,
                                 self._output_image_chooser, self._output_protocol_chooser]

        self._concatenate_button = ttk.Button(self._tab, text='Concatenate', command=self._apply_concatenate,
                                                       state='disabled')

    def get_tab(self):
        row = 0
        label = ttk.Label(self._tab, text="Concatenate shells", font=(None, 14))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1
        label = ttk.Label(self._tab, text="Concatenate multiple aligned 4d image series of the same subject. ",
                          font=(None, 9, 'italic'))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1
        self._image_1_chooser.render(row)
        row += 1
        self._protocol_1_chooser.render(row)
        row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(8, 3))
        row += 1
        self._image_2_chooser.render(row)
        row += 1
        self._protocol_2_chooser.render(row)
        row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(8, 3))
        row += 1
        self._output_image_chooser.render(row)
        row += 1
        self._output_protocol_chooser.render(row)
        row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(10, 3))
        row += 1
        self._concatenate_button.grid(row=row, sticky='W', pady=(10, 0))
        return self._tab

    def _onchange_concat(self, cb_name):
        if (cb_name == 'image_1_chooser' or cb_name == 'image_2_chooser') \
                and self._image_1_chooser.is_valid() and self._image_2_chooser.is_valid():

            img1_name = os.path.splitext(self._image_1_chooser.get_value())[0]
            img1_name = img1_name.replace('.nii', '')

            img2_name = os.path.splitext(self._image_2_chooser.get_value())[0]
            img2_name = os.path.basename(img2_name.replace('.nii', ''))

            self._output_image_chooser.initial_file = img1_name + '_' + img2_name + '.nii.gz'

        if (cb_name == 'protocol_1_chooser' or cb_name == 'protocol_2_chooser')\
                and self._protocol_1_chooser.is_valid() and self._protocol_2_chooser.is_valid():

            prtcl_1_name = os.path.splitext(self._protocol_1_chooser.get_value())[0]
            prtcl_2_name = os.path.basename(os.path.splitext(self._protocol_2_chooser.get_value())[0])

            self._output_protocol_chooser.initial_file = prtcl_1_name + '_' + prtcl_2_name + '.prtcl'

        all_valid = True
        for field in self._validate_fields:
            if not field.is_valid():
                all_valid = False
                break

        if all_valid:
            self._concatenate_button.config(state='normal')
        else:
            self._concatenate_button.config(state='disabled')

    def _apply_concatenate(self):
        self._concatenate_button.config(state='disabled')
        img1 = self._image_1_chooser.get_value()
        img2 = self._image_2_chooser.get_value()
        protocol1 = self._protocol_1_chooser.get_value()
        protocol2 = self._protocol_2_chooser.get_value()
        output_protocol = self._output_protocol_chooser.get_value()
        output_img = self._output_image_chooser.get_value()

        items = ({'volume': img1, 'protocol': protocol1}, {'volume': img2, 'protocol': protocol2})
        concatenate_mri_sets(items, output_img, output_protocol)

        tkMessageBox.showinfo('Action completed', 'The action has been completed.')
        self._concatenate_button.config(state='normal')


class ViewResultsTab(TabContainer):

    def __init__(self, window):
        super(ViewResultsTab, self).__init__(window, 'View results')

        self._dimensions = {}
        self._dimension_shape = []
        self._parameter_files = {}

        self._input_dir = DirectoryBrowserWidget(
            self._tab,
            lambda trc1, trc2, trc3:
            self._onchange_cb('input_dir_chooser'),
            True,
            'Select input folder: ',
            '(Choose a result directory)')

        self._roi_dimension_index = DropdownWidget(
            self._tab,
            lambda val: self._onchange_cb('roi_dimension_chooser'),
            (),
            '',
            'Select dimension: ',
            '(Select an input dir first)')

        self._roi_slice_index = ListboxWidget(
            self._tab,
            lambda val: self._onchange_cb('roi_slice_chooser'),
            (),
            '',
            BROWSE,
            1,
            'Select slice: ',
            '(Select an input dir first)')

        self._maps_chooser = ListboxWidget(
            self._tab,
            lambda val: self._onchange_cb('maps_chooser'),
            (),
            '',
            EXTENDED,
            10,
            'Select parameters: ',
            '(Select the parameters\n you would like to display)')

        self._validate_fields = [self._input_dir, self._roi_dimension_index, self._roi_slice_index]

        self._view_slices_button = ttk.Button(self._tab, text='View', command=self._view_slices,
                                              state='disabled')

    def get_tab(self):
        row = 0
        label = ttk.Label(self._tab, text="View results", font=(None, 14))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1

        label = ttk.Label(self._tab, text="View all maps in a given folder.",
                          font=(None, 9, 'italic'))
        label.grid(row=row, column=0, columnspan=4, sticky=W)
        row += 1

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(5, 3))
        row += 1
        self._input_dir.render(row)
        row += 1
        self._roi_dimension_index.render(row)
        row += 1
        self._roi_slice_index.render(row)
        row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(8, 3))
        row += 1
        self._maps_chooser.render(row)
        row += 1
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=row, columnspan=5, sticky="EW", pady=(10, 3))
        row += 1
        self._view_slices_button.grid(row=row, sticky='W', pady=(10, 0))
        return self._tab

    def _onchange_cb(self, cb_name):
        if cb_name == 'input_dir_chooser':
            folder = self._input_dir.get_value()
            result_files = glob.glob(os.path.join(folder, '*.nii.gz')) + glob.glob(os.path.join(folder, '*.nii'))

            a_file = None
            self._parameter_files = {}
            for f in result_files:
                name = os.path.splitext(os.path.basename(f))[0]
                name = name.replace('.nii', '')
                self._parameter_files.update({name: name})
                a_file = f

            items_list = sorted(self._parameter_files.keys())
            selected_items = filter(lambda v: all(m not in v for m in ('eig', '.d', '.sigma')), items_list)
            self._maps_chooser.set_items(sorted(self._parameter_files.keys()), default_items=selected_items)

            if a_file:
                data = nib.load(a_file)
                shape = data.shape
                if len(shape) > 3:
                    shape = list(shape)
                    del shape[-1]

                if self._dimension_shape != shape:
                    self._dimension_shape = shape
                    self._dimensions = {}
                    items = []
                    for i in range(len(shape)):
                        items.append('{} ({} items)'.format(i, shape[i]))
                        self._dimensions.update({'{} ({} items)'.format(i, shape[i]): i})
                    self._roi_dimension_index.set_items(items, items[0])
                    self._update_roi_slice_chooser()

        elif cb_name == 'roi_dimension_chooser':
            self._update_roi_slice_chooser()

        all_valid = True
        for field in self._validate_fields:
            if not field.is_valid():
                all_valid = False
                break

        if all_valid:
            self._view_slices_button.config(state='normal')
        else:
            self._view_slices_button.config(state='disabled')

    def _update_roi_slice_chooser(self):
        dimension = self._dimensions[self._roi_dimension_index.get_value()]
        length = self._dimension_shape[dimension]
        self._roi_slice_index.set_items(range(length), int(length/2))

    def _view_slices(self):
        slice_dimension = int(self._dimensions[self._roi_dimension_index.get_value()])
        slice_index = int(self._roi_slice_index.get_value()[0])

        params_list = []
        selected_maps = self._maps_chooser.get_value()

        if not selected_maps:
            for key, value in self._parameter_files.items():
                params_list.append(value)
        else:
            for ind in selected_maps:
                params_list.append(self._parameter_files.get(ind))

        def results_slice_cb():
            view_results_slice(self._input_dir.get_value(), slice_dimension, slice_index, maps_to_show=params_list)

        view_process = multiprocessing.Process(target=results_slice_cb, args=())
        view_process.start()


class CompositeWidget(object):

    def __init__(self, root_window, onchange_cb):
        super(CompositeWidget, self).__init__()
        self._root_window = root_window
        self._onchange_cb = onchange_cb

    def is_valid(self):
        pass

    def get_value(self):
        pass

    def render(self, row):
        pass


class FileBrowserWidget(CompositeWidget):

    OPEN = 1
    SAVE = 2

    def __init__(self, root_window, onchange_cb, file_types, label_text, helper_text, dialog_type=None):
        super(FileBrowserWidget, self).__init__(root_window, onchange_cb)

        self._initial_file = None

        if not dialog_type:
            self._dialog_type = FileBrowserWidget.OPEN
        else:
            self._dialog_type = dialog_type

        self._file_types = file_types
        self._label_text = label_text
        self._helper_text = helper_text

        self._browse_button = ttk.Button(self._root_window, text='Browse', command=self._get_filename)
        self._fname_var = StringVar()
        self._fname_var.trace('w', self._onchange_cb)
        self._fname_entry = ttk.Entry(self._root_window, textvariable=self._fname_var, width=30)

    @property
    def initial_file(self):
        return self._initial_file

    @initial_file.setter
    def initial_file(self, value):
        self._initial_file = value
        self._fname_var.set(value)

    def is_valid(self):
        return self._fname_var.get()

    def get_value(self):
        return self._fname_var.get()

    def render(self, row):
        ttk.Label(self._root_window, text=self._label_text).grid(row=row, column=0, padx=(0, 5), pady=(5, 0), sticky=E)
        self._browse_button.grid(row=row, column=1, padx=(0, 3), pady=(5, 0), sticky=W)
        self._fname_entry.grid(row=row, column=2, pady=(5, 0), sticky=W)
        ttk.Label(self._root_window, text=self._helper_text, font=(None, 9, 'italic')).grid(row=row, column=3,
                                                                                            padx=(12, 0), sticky=W,
                                                                                            pady=(5, 0))

    def _get_filename(self):
        self.file_opt = options = {}
        options['defaultextension'] = ''
        options['filetypes'] = self._file_types

        init_dir = self._fname_entry.get()
        if not init_dir:
            init_dir = self.initial_file

        if init_dir:
            options['initialdir'] = os.path.dirname(init_dir)
            options['initialfile'] = os.path.basename(init_dir)

        options['parent'] = self._root_window
        options['title'] = self._label_text

        if self._dialog_type == self.OPEN:
            filename = tkFileDialog.askopenfilename(**options)
        else:
            filename = tkFileDialog.asksaveasfilename(**options)
        if filename:
            self._fname_var.set(filename)

    @staticmethod
    def common_file_types(choice):
        if choice == 'image_volumes':
            l = (('All image files', ('*.nii', '*.nii.gz', '*.img')),
                 ('Nifti files', ('*.nii', '*.nii.gz')),
                 ('Analyze files', '*.img'),
                 ('All files', '.*'))
        elif choice == 'protocol_files':
            l = (('Protocol files', '*.prtcl'),
                 ('All files', '.*'))
        elif choice == 'txt':
            l = (('Text files', '*.txt'),
                 ('All files', '.*'))
        elif choice == 'bvec':
            l = (('b-vector files', '*.bvec'),
                 ('Text files', '*.txt'),
                 ('All files', '.*'))
        elif choice == 'bval':
            l = (('b-values files', '*.bval'),
                 ('Text files', '*.txt'),
                 ('All files', '.*'))
        else:
            return ('All files', '.*'),

        if platform.system() == 'Windows':
            return list(reversed(l))
        return l


class DirectoryBrowserWidget(CompositeWidget):

    def __init__(self, root_window, onchange_cb, must_exist, label_text, helper_text):
        super(DirectoryBrowserWidget, self).__init__(root_window, onchange_cb)

        self._initial_dir = None
        self.must_exist = must_exist

        self._label_text = label_text
        self._helper_text = helper_text

        self._browse_button = ttk.Button(self._root_window, text='Browse', command=self._get_dir_name)
        self._fname_var = StringVar()
        self._fname_var.trace('w', self._onchange_cb)
        self._fname_entry = ttk.Entry(self._root_window, textvariable=self._fname_var, width=30)

    @property
    def initial_dir(self):
        return self._initial_dir

    @initial_dir.setter
    def initial_dir(self, directory):
        if self.must_exist:
            if os.path.isdir(directory):
                self._initial_dir = directory
                self._fname_var.set(self._initial_dir)
        else:
            self._initial_dir = directory
            self._fname_var.set(self._initial_dir)

    def is_valid(self):
        if self.must_exist:
            v = self._fname_var.get()
            return os.path.isdir(v)
        return True

    def get_value(self):
        return self._fname_var.get()

    def render(self, row):
        ttk.Label(self._root_window, text=self._label_text).grid(row=row, column=0, padx=(0, 5), pady=(5, 0), sticky=E)
        self._browse_button.grid(row=row, column=1, padx=(0, 3), pady=(5, 0), sticky=W)
        self._fname_entry.grid(row=row, column=2, pady=(5, 0), sticky=W)
        ttk.Label(self._root_window, text=self._helper_text, font=(None, 9, 'italic')).grid(row=row, column=3,
                                                                                            padx=(12, 0), sticky=W,
                                                                                            pady=(5, 0))

    def _get_dir_name(self):
        self.file_opt = options = {}
        options['mustexist'] = self.must_exist

        init_dir = self._fname_entry.get()
        if init_dir:
            options['initialdir'] = init_dir
        elif self.initial_dir:
            options['initialdir'] = self.initial_dir

        options['parent'] = self._root_window
        options['title'] = self._label_text

        filename = tkFileDialog.askdirectory(**options)
        if filename:
            self._fname_var.set(filename)


class TextboxWidget(CompositeWidget):

    def __init__(self, root_window, onchange_cb, label_text, helper_text, default_val='',
                 required=False, state='normal'):
        super(TextboxWidget, self).__init__(root_window, onchange_cb)

        self._required = required

        self._label_text = label_text
        self._helper_text = helper_text

        self._fname_var = StringVar()
        self._fname_var.set(default_val)
        self._fname_var.trace('w', self._onchange_cb)
        self._fname_entry = ttk.Entry(self._root_window, textvariable=self._fname_var, width=30, state=state)

    def set_state(self, state):
        self._fname_entry.config(state=state)

    def set_value(self, value):
        self._fname_var.set(value)

    def is_valid(self):
        if self._required:
            return self._fname_var.get() is not None
        return True

    def get_value(self):
        return self._fname_var.get()

    def render(self, row):
        ttk.Label(self._root_window, text=self._label_text).grid(row=row, column=0, padx=(0, 5), pady=(5, 0), sticky=E)
        self._fname_entry.grid(row=row, column=1, columnspan=2, pady=(5, 0), sticky='EW')
        ttk.Label(self._root_window, text=self._helper_text, font=(None, 9, 'italic')).grid(row=row, column=3,
                                                                                            padx=(12, 0), sticky=W,
                                                                                            pady=(5, 0))


class DropdownWidget(CompositeWidget):

    def __init__(self, root_window, onchange_cb, items, default_item, label_text, helper_text):
        super(DropdownWidget, self).__init__(root_window, onchange_cb)

        self._label_text = label_text
        self._helper_text = helper_text

        self._chooser_var = StringVar(self._root_window)
        self._chooser = ttk.OptionMenu(self._root_window, self._chooser_var, default_item, *items, command=onchange_cb)

    def set_items(self, items, default_item):
        self._chooser_var.set('')
        self._chooser['menu'].delete(0, 'end')

        for choice in items:
            def func(tmp=choice, cb=self._onchange_cb):
                self._chooser.setvar(self._chooser.cget("textvariable"), value=tmp)
                cb(tmp)
            self._chooser['menu'].add_command(label=choice, command=func)
        self._chooser_var.set(default_item)

    def is_valid(self):
        return True

    def get_value(self):
        return self._chooser_var.get()

    def render(self, row):
        ttk.Label(self._root_window, text=self._label_text).grid(row=row, column=0, padx=(0, 5), pady=(5, 0), sticky=E)
        self._chooser.grid(row=row, column=1, columnspan=2, padx=(0, 3), pady=(5, 0), sticky='WE')
        ttk.Label(self._root_window, text=self._helper_text, font=(None, 9, 'italic')).grid(row=row, column=3,
                                                                                            padx=(12, 0), sticky=W,
                                                                                            pady=(5, 0))


class ListboxWidget(CompositeWidget):

    def __init__(self, root_window, onchange_cb, items, default_item, selectmode, height, label_text, helper_text):
        super(ListboxWidget, self).__init__(root_window, onchange_cb)

        self._selectbox_height = height
        self._label_text = label_text
        self._helper_text = helper_text

        self._chooser_frame = ttk.Frame(self._root_window)

        self._scrollbar = ttk.Scrollbar(self._chooser_frame, orient=VERTICAL)
        self._chooser = Listbox(self._chooser_frame, selectmode=selectmode, exportselection=0,
                                yscrollcommand=self._mouse_scroll_detect, height=height)
        self._chooser.bind('<<ListboxSelect>>', onchange_cb)
        self._scrollbar.config(command=self._scrolling)

        if items:
            self.set_items(items, default_item)

        self._chooser_frame.grid_columnconfigure(0, weight=1)
        self._chooser.grid(row=0, column=0, sticky='EW')
        self._scrollbar.grid(row=0, column=1, sticky='ENS')

    def set_items(self, items, default_item=None, default_items=None):
        self._chooser.delete(0, END)
        for e in items:
            self._chooser.insert(END, e)

        if default_item == 'ALL':
            self._chooser.selection_set(0, END)

        if default_item in items:
            self._chooser.selection_set(items.index(default_item))
            self._chooser.see(items.index(default_item))

        if default_items:
            for item in default_items:
                self._chooser.selection_set(items.index(item))

    def is_valid(self):
        selection = self._chooser.curselection()
        if selection:
            return True
        return False

    def get_value(self):
        selection = self._chooser.curselection()
        items = []
        for ind in selection:
            items.append(self._chooser.get(ind))
        return items

    def render(self, row):
        ttk.Label(self._root_window, text=self._label_text).grid(row=row, column=0, padx=(0, 5), pady=(5, 0), sticky=E)
        self._chooser_frame.grid(row=row, column=1, columnspan=2, padx=(0, 3), pady=(5, 0), sticky='WE')
        ttk.Label(self._root_window, text=self._helper_text, font=(None, 9, 'italic')).grid(row=row, column=3,
                                                                                            padx=(12, 0), sticky=W,
                                                                                            pady=(5, 0))

    def _mouse_scroll_detect(self, *args):
        self._scrollbar.set(*args)

        if self._selectbox_height == 1:
            index = self._chooser.nearest(int(self._chooser.yview()[0] * 10))
            self._chooser.select_clear(0, END)
            self._chooser.selection_set(index, index)
        self._onchange_cb(None)

    def _scrolling(self, *args):
        apply(self._chooser.yview, args)
        self._onchange_cb(None)


class YesNonWidget(CompositeWidget):

    def __init__(self, root_window, onchange_cb, label_text, helper_text, default_val=0):
        super(YesNonWidget, self).__init__(root_window, onchange_cb)

        self._label_text = label_text
        self._helper_text = helper_text

        self._chooser_var = IntVar(self._root_window)
        self._yes = ttk.Radiobutton(self._root_window, text='Yes', variable=self._chooser_var,
                                    value=1, command=onchange_cb)
        self._no = ttk.Radiobutton(self._root_window, text='No', variable=self._chooser_var,
                                   value=0, command=onchange_cb)
        self._chooser_var.set(default_val)

    def is_valid(self):
        return True

    def get_value(self):
        return self._chooser_var.get() == 1

    def render(self, row):
        ttk.Label(self._root_window, text=self._label_text).grid(row=row, column=0, padx=(0, 5), pady=(5, 0), sticky=E)
        self._yes.grid(row=row, column=1, padx=(0, 3), pady=(5, 0), sticky='WE')
        self._no.grid(row=row, column=2, padx=(0, 3), pady=(5, 0), sticky='WE')
        ttk.Label(self._root_window, text=self._helper_text, font=(None, 9, 'italic')).grid(row=row, column=3,
                                                                                            padx=(12, 0), sticky=W,
                                                                                            pady=(5, 0))


class SubWindowWidget(CompositeWidget):

    def __init__(self, root_window, subwindow, label_text, helper_text):
        super(SubWindowWidget, self).__init__(root_window, None)
        self._subwindow = subwindow
        self._label_text = label_text
        self._helper_text = helper_text
        self._launch_button = ttk.Button(self._root_window, text='Launch options frame', command=self._launch_frame)

    def render(self, row):
        ttk.Label(self._root_window, text=self._label_text).grid(row=row, column=0, padx=(0, 5), pady=(5, 0), sticky=E)
        self._launch_button.grid(row=row, column=1, columnspan=2, padx=(0, 3), pady=(5, 0), sticky=W)
        ttk.Label(self._root_window, text=self._helper_text, font=(None, 9, 'italic')).grid(row=row, column=3,
                                                                                            padx=(12, 0), sticky=W,
                                                                                            pady=(5, 0))

    def _launch_frame(self):
        t = Tkinter.Toplevel(self._root_window)
        self._subwindow.render(t)
        t.wm_title(self._subwindow.window_title)