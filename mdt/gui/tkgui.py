from Tkconstants import W, E, BOTH, FALSE, YES, HORIZONTAL, VERTICAL, END, EXTENDED, BROWSE, INSERT
from Tkinter import Tk, StringVar, IntVar, Listbox, TclError
import Tkinter
import copy
from functools import wraps
import glob
import numbers
import os
import threading
import tkFileDialog
import tkMessageBox
import ttk
import sys
import multiprocessing
from math import log10
import platform
from nibabel.spatialimages import ImageFileError
from numpy import genfromtxt
import nibabel as nib
import numpy as np
import mdt
from mdt.gui.utils import print_welcome_message, update_user_settings, split_image_path, IntegerGenerator, OptimOptions
from mdt.log_handlers import LogListenerInterface, LogDispatchHandler
import mdt.utils
from mdt import load_dwi, load_brain_mask, create_median_otsu_brain_mask, \
    create_slice_roi, load_protocol_bval_bvec, view_results_slice, concatenate_mri_sets
from mdt.visualization import MapsVisualizer
import mdt.protocols
import mdt.configuration
from mot.factory import get_optimizer_by_name

__author__ = 'Robbert Harms'
__date__ = "2014-10-01"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


mdt.utils.setup_logging(disable_existing_loggers=True)


def get_window():
    return ToolkitGUIWindow()


class ToolkitGUIWindow(Tk):

    def __init__(self):
        Tk.__init__(self)

        s = ttk.Style()
        try:
            s.theme_use('clam')
        except TclError:
            pass

        self.wm_title("Diffusion MRI Toolbox")
        self.resizable(width=FALSE, height=FALSE)
        self.update_idletasks()
        width = 900
        height = 630
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        notebook = MainNotebook(self)
        notebook.pack(fill=BOTH, expand=YES)

        txt_frame = ttk.Frame(self)
        self._log_box = ScrolledText(txt_frame, wrap='none')
        self._log_box.pack(fill=BOTH, expand=YES)
        txt_frame.pack(fill=BOTH, expand=YES)

        stdout_old = sys.stdout
        stderr_old = sys.stderr

        self.after(100, self._window_start_cb)

        def on_closing():
            sys.stdout = stdout_old
            sys.stderr = stderr_old
            self.destroy()

        self.protocol("WM_DELETE_WINDOW", on_closing)

    def _window_start_cb(self):
        log_listener = TextboxLogListener(self._log_box)
        sys.stdout = log_listener
        sys.stderr = log_listener

        LogDispatchHandler.add_listener(log_listener)
        print_welcome_message()
        update_user_settings()


class MainNotebook(ttk.Notebook):

    def __init__(self, window):
        ttk.Notebook.__init__(self, window)

        self.tabs = [RunModelTab(window),
                     GenerateBrainMaskTab(window),
                     GenerateROIMaskTab(window),
                     GenerateProtocolFileTab(window),
                     ConcatenateShellsTab(window),
                     ViewResultsTab(window)]

        for tab in self.tabs:
            self.add(tab.get_tab(), text=tab.tab_name)

        self.bind_all("<<NotebookTabChanged>>", self._tab_change_cb)

    def _tab_change_cb(self, event):
        self.tabs[event.widget.index("current")].tab_selected()


class TextboxLogListener(LogListenerInterface):

    def __init__(self, text_box):
        self.text_box = text_box

    def emit(self, record, formatted_message):
        self.write(formatted_message)
        self.write("\n")

    def write(self, string):
        self.text_box.configure(state='normal')
        self.text_box.insert(END, string)
        self.text_box.configure(state='disabled')
        self.text_box.see(END)

    def flush(self):
        pass


class TabContainer(object):

    last_used_dwi_image = None
    last_used_mask = None
    last_used_roi_mask = None
    last_used_protocol = None
    last_used_model_output_folder = None
    last_optimized_model = None

    def __init__(self, window, tab_name):
        self.window = window
        self.tab_name = tab_name
        self._tab = ttk.Frame()
        self._tab.config(padding=(10, 13, 10, 13))
        self._tab.grid_columnconfigure(3, weight=1)

    def get_tab(self):
        """Get the tab frame for this tab object"""

    def tab_selected(self):
        """Called by the Notebook on the moment this tab becomes active."""

    def _init_path_chooser(self, file_chooser, path):
        """Init one of the file/directory choosers with the given path.

        If path is None or does not exist, it is not set. This uses the property file_chooser.initial_file
        to set the path.

        Args:
            file_chooser: the file chooser to set the path of
            path (str): the path to set as initial file
        """
        if path and os.path.exists(path):
            file_chooser.initial_file = path

    @staticmethod
    def message_decorator(header, footer):
        """This creates and returns a decorator that prints a header and footer before executing the function.

        Args:
            header (str): the header text, we will add extra decoration to it
            foot (str): the footer text, we will add extra decoration to it

        Returns:
            decorator function
        """
        def _called_decorator(dec_func):

            @wraps(dec_func)
            def _decorator(*args, **kwargs):
                print('')
                print(header)
                print('-'*20)

                response = dec_func(*args, **kwargs)

                print('-'*20)
                print(footer)

                return response
            return _decorator

        return _called_decorator


class RunModelTab(TabContainer):

    def __init__(self, window):
        super(RunModelTab, self).__init__(window, 'Run model')

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
        next_row = IntegerGenerator()

        label = ttk.Label(self._tab, text="Run model", font=(None, 14))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="Optimize a model to your data.",
                          font=(None, 9, 'italic'))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        for field in self._io_fields:
            field.render(next_row())
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(8, 3))

        for field in self._model_select_fields:
            field.render(next_row())

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(8, 3))
        self._optim_options_button.render(next_row())

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(8, 3))
        self._run_button.grid(row=next_row(), sticky='W', pady=(10, 0))

        return self._tab

    def tab_selected(self):
        self._init_path_chooser(self._image_vol_chooser, TabContainer.last_used_dwi_image)
        self._init_path_chooser(self._brain_mask_chooser, TabContainer.last_used_mask)
        self._init_path_chooser(self._brain_mask_chooser, TabContainer.last_used_roi_mask)
        self._init_path_chooser(self._protocol_file_chooser, TabContainer.last_used_protocol)

    def _run_model(self):
        self._test_protocol()
        thr = threading.Thread(target=self._run_optimizer)
        thr.start()

    def _test_protocol(self):
        model_name = self._model_select_chooser.get_value()
        model = mdt.get_model(model_name)

        protocol = mdt.load_protocol(self._protocol_file_chooser.get_value())
        protocol_sufficient = model.is_protocol_sufficient(protocol)

        if not protocol_sufficient:
            problems = model.get_protocol_problems(protocol)
            tkMessageBox.showerror('Protocol insufficient', "\n".join(['- ' + str(p) for p in problems]))

    @TabContainer.message_decorator('Starting model fitting, please wait.',
                                    'Finished model fitting. You can view the results using the "View results" tab.')
    def _run_optimizer(self):
        self._run_button.config(state='disabled')

        self._set_last_run_settings()

        optimizer = self.optim_options.get_optimizer()
        model_name = self._model_select_chooser.get_value()

        mdt.fit_model(model_name,
                      self._image_vol_chooser.get_value(),
                      self._protocol_file_chooser.get_value(),
                      self._brain_mask_chooser.get_value(),
                      self._output_dir_chooser.get_value(),
                      optimizer=optimizer,
                      recalculate=True,
                      only_recalculate_last=self.optim_options.recalculate_all)

        self._run_button.config(state='normal')

    def _set_last_run_settings(self):
        TabContainer.last_used_dwi_image = self._image_vol_chooser.get_value()
        TabContainer.last_used_protocol = self._protocol_file_chooser.get_value()
        TabContainer.last_used_model_output_folder = self._output_dir_chooser.get_value()
        TabContainer.last_optimized_model = self._model_select_chooser.get_value()

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key

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


class SubWindow(object):

    def __init__(self, window_title):
        self._window_title = window_title

    @property
    def window_title(self):
        return self._window_title

    def render(self, toplevel):
        """Render the items in the given frame"""


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

        self._patience_box = TextboxWidget(
            subframe,
            'patience_box',
            self._onchange_cb,
            'Patience: ',
            '(The amount of iterations to wait per parameter)',
            default_val=self._optim_options.patience)

        self._recalculate_all = YesNonWidget(
            subframe,
            'recalculate_all',
            self._onchange_cb,
            'Recalculate all: ',
            '(If yes, recalculate all maps, if no, only the last.\n This is only for cascades.)',
            default_val=self._optim_options.recalculate_all)

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
        cl_environment_names = self._optim_options.cl_environments.keys()
        for ind in self._optim_options.cl_envs_indices:
            default_cl_environments.append(cl_environment_names[ind])

        self._devices_chooser.set_items(self._optim_options.cl_environments.keys(),
                                        default_items=default_cl_environments)

        fields = [self._optim_routine_chooser, self._patience_box, self._recalculate_all, self._extra_optim_runs,
                  self._smoothing_routine_chooser, self._smoothing_size, self._devices_chooser]

        button_frame = self._get_button_frame(subframe, window)

        next_row = IntegerGenerator()
        label = ttk.Label(subframe, text="Optimization options", font=(None, 14))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        label = ttk.Label(subframe, text="Options for the optimization routines, these are advanced settings.",
                          font=(None, 9, 'italic'))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        for field in fields:
            field.render(next_row())

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        button_frame.grid(row=next_row(), sticky='W', pady=(10, 0), columnspan=4)

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

        elif id_key == 'cl_environment_chooser':
            chosen_keys = self._devices_chooser.get_value()
            l = [ind for ind, key in enumerate(self._optim_options.cl_environments.keys()) if key in chosen_keys]
            if l:
                self._optim_options.cl_envs_indices = l


class GenerateBrainMaskTab(TabContainer):

    def __init__(self, window):
        super(GenerateBrainMaskTab, self).__init__(window, 'Generate brain mask')

        self._image_vol_chooser = FileBrowserWidget(
            self._tab, 'image_vol_chooser', self._onchange_cb,
            FileBrowserWidget.common_file_types('image_volumes'), 'Select 4d image: ',
            '(Select the measured 4d diffusion weighted image)')

        self._protocol_file_chooser = FileBrowserWidget(
            self._tab, 'protocol_file_chooser', self._onchange_cb,
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select protocol file: ', '(To create one see the tab "Generate protocol file")')

        self._output_bm_chooser = FileBrowserWidget(
            self._tab, 'output_bm_chooser', self._onchange_cb,
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select output file: ', '(Default is <volume_name>_mask.nii.gz)', dialog_type=FileBrowserWidget.SAVE)

        self._median_radius_box = TextboxWidget(
            self._tab, 'median_radius_box', self._onchange_cb,
            'Median radius: ', '(Radius (in voxels) of the applied median filter)', default_val='4')

        self._numpasses_box = TextboxWidget(
            self._tab, 'numpasses_box', self._onchange_cb,
            'Number of passes: ', '(Number of median filter passes)', default_val='4')

        self._threshold_box = TextboxWidget(
            self._tab, 'threshold', self._onchange_cb,
            'Final threshold: ', '(Additional masking threshold)', default_val='0')

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
        next_row = IntegerGenerator()

        label = ttk.Label(self._tab, text="Generate brain mask", font=(None, 14))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="Creates a mask for a whole brain using the median-otsu algorithm.",
                          font=(None, 9, 'italic'))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)
        
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        
        for field in self._mask_items:
            field.render(next_row())
            
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        
        self._whole_mask_buttons_frame.grid(row=next_row(), sticky='W', pady=(10, 0), columnspan=4)
        return self._tab

    def tab_selected(self):
        self._init_path_chooser(self._output_bm_chooser, TabContainer.last_used_mask)
        self._init_path_chooser(self._image_vol_chooser, TabContainer.last_used_dwi_image)
        self._init_path_chooser(self._protocol_file_chooser, TabContainer.last_used_protocol)

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key
        if id_key == 'image_vol_chooser':
            path, img_name = split_image_path(calling_widget.get_value())[0:2]
            mask_name = os.path.join(path, img_name + '_mask.nii.gz')
            self._output_bm_chooser.initial_file = mask_name

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
        mask_create_thread = threading.Thread(target=self._create_brain_mask)
        mask_create_thread.start()

    @TabContainer.message_decorator('Started creating a mask.', 'Finished creating a mask')
    def _create_brain_mask(self):
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

        create_median_otsu_brain_mask(volume, prtcl, output_fname, median_radius=filter_mean_radius,
                                      numpass=filter_passes, mask_threshold=threshold)

        TabContainer.last_used_dwi_image = volume
        TabContainer.last_used_protocol = prtcl
        TabContainer.last_used_mask = output_fname

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
            'brain_mask_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select brain mask: ', '(To create one see the tab "Generate brain mask")')

        self._roi_dimension_index = DropdownWidget(
            self._tab,
            'roi_dimension_chooser',
            self._onchange_cb,
            (),
            '',
            'Select dimension: ',
            '(Select a brain mask first)')

        self._roi_slice_index = ListboxWidget(
            self._tab,
            'roi_slice_chooser',
            self._onchange_cb,
            (),
            '',
            BROWSE,
            1,
            'Select slice: ',
            '(Select a brain mask first)')

        self._output_roi_chooser = FileBrowserWidget(
            self._tab,
            'roi_output_chooser',
            self._onchange_cb,
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
        next_row = IntegerGenerator()

        label = ttk.Label(self._tab, text="Generate ROI mask", font=(None, 14))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)
        
        label = ttk.Label(self._tab, text="Create a brain mask with a Region Of Interest "
                                          "that only includes the voxels in the selected slice. ",
                          font=(None, 9, 'italic'))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)
        
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        
        for field in self._roi_items:
            field.render(next_row())
            
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(10, 3))
        
        self._buttons_frame.grid(row=next_row(), sticky='W', pady=(10, 0), columnspan=4)
        return self._tab

    def tab_selected(self):
        self._init_path_chooser(self._output_roi_chooser, TabContainer.last_used_roi_mask)
        self._init_path_chooser(self._brain_mask_vol_chooser, TabContainer.last_used_mask)

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key
        brain_mask_fname = self._brain_mask_vol_chooser.get_value()

        if brain_mask_fname:
            if id_key == 'brain_mask_chooser':
                try:
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
                except ImageFileError:
                    tkMessageBox.showerror('File not an image', 'The selected file could not be loaded.')
                    return

            elif id_key == 'roi_dimension_chooser':
                self._update_roi_slice_chooser()

            brain_mask = self._brain_mask_vol_chooser.get_value()
            roi_dim = int(self._dimensions[self._roi_dimension_index.get_value()])
            roi_slice = int(self._roi_slice_index.get_value()[0])
            mask_name = os.path.splitext(os.path.basename(brain_mask))[0].replace('.nii', '')
            output_fname = os.path.join(os.path.dirname(brain_mask), mask_name + '_' + repr(roi_dim) + '_' +
                                        repr(roi_slice)) + '.nii.gz'
            self._output_roi_chooser.initial_file = output_fname

        all_valid = all(field.is_valid() for field in self._roi_items)
        if all_valid:
            self._run_slice_roi_button.config(state='normal')
        else:
            self._run_slice_roi_button.config(state='disabled')

    def _update_roi_slice_chooser(self):
        dimension = self._dimensions[self._roi_dimension_index.get_value()]
        length = self._dimension_shape[dimension]
        self._roi_slice_index.set_items(range(length), int(length/2))

    @TabContainer.message_decorator('Generating a slice ROI', 'Generating a slice ROI')
    def _run_slice_roi(self):
        self._run_slice_roi_button.config(state='disabled')
        print('please wait...')

        brain_mask = self._brain_mask_vol_chooser.get_value()
        roi_dim = int(self._dimensions[self._roi_dimension_index.get_value()])
        roi_slice = int(self._roi_slice_index.get_value()[0])
        output_fname = self._output_roi_chooser.get_value()
        create_slice_roi(brain_mask, roi_dim, roi_slice, output_fname, overwrite_if_exists=True)

        TabContainer.last_used_mask = brain_mask
        TabContainer.last_used_roi_mask = output_fname

        self._run_slice_roi_button.config(state='normal')


class GenerateProtocolFileTab(TabContainer):

    def __init__(self, window):
        super(GenerateProtocolFileTab, self).__init__(window, 'Generate protocol file')

        self._extra_options_window = ProtocolExtraOptionsWindow(self)

        self._bvec_chooser = FileBrowserWidget(
            self._tab,
            'bvec_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('bvec'),
            'Select bvec file: ', '(Select the gradient vectors file)',
            dialog_type=FileBrowserWidget.OPEN)

        self._bval_chooser = FileBrowserWidget(
            self._tab,
            'bval_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('bval'),
            'Select bval file: ', '(Select the gradient b-values file)',
            dialog_type=FileBrowserWidget.OPEN)

        self._bval_scale_box = TextboxWidget(
            self._tab,
            'bval_scale_box',
            self._onchange_cb,
            'B-value scale factor: ', '(We expect the the b-values in the\nresult protocol in s/m^2)',
            default_val='1e6')

        self._output_protocol_chooser = FileBrowserWidget(
            self._tab,
            'output_protocol_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select output (protocol) file: ', '(Default is <bvec_name>.prtcl)', dialog_type=FileBrowserWidget.SAVE)

        self._extra_options_button = SubWindowWidget(
            self._tab,
            'extra_options_button',
            self._extra_options_window,
            'Extra options: ',
            '(Add additional columns, some models need them)')

        self._to_protocol_items = (self._bvec_chooser, self._bval_chooser, self._bval_scale_box,
                                   self._output_protocol_chooser)

        self._buttons_frame = ttk.Frame(self._tab)
        self._generate_prtcl_button = ttk.Button(self._buttons_frame, text='Generate protocol',
                                                 command=self._generate_protocol, state='disabled')

        self._view_results_button = ttk.Button(self._buttons_frame, text='View protocol file',
                                               command=self._view_results, state='disabled')
        self._generate_prtcl_button.grid(row=0)
        self._view_results_button.grid(row=0, column=1, padx=(10, 0))

    def get_tab(self):
        next_row = IntegerGenerator()
        label = ttk.Label(self._tab, text="Generate protocol file", font=(None, 14))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="Create a protocol file containing all your sequence information.",
                          font=(None, 9, 'italic'))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        for field in self._to_protocol_items:
            field.render(next_row())

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(8, 3))
        self._extra_options_button.render(next_row())

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(10, 3))
        self._buttons_frame.grid(row=next_row(), sticky='W', columnspan=4, pady=(10, 0))
        return self._tab

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key

        if id_key == 'bvec_chooser':
            self._output_protocol_chooser.initial_file = os.path.splitext(calling_widget.get_value())[0] + '.prtcl'

        if id_key == 'bval_chooser':
            bval_file = calling_widget.get_value()
            if os.path.isfile(bval_file):
                data = genfromtxt(bval_file)
                try:
                    if log10(np.mean(data)) >= 8:
                        self._bval_scale_box.set_value('1')
                    else:
                        self._bval_scale_box.set_value('1e6')
                except ValueError:
                    self._bval_scale_box.set_value('1e6')

        if self._output_protocol_chooser.is_valid():
            if os.path.isfile(calling_widget.get_value()):
                self._view_results_button.config(state='normal')
            else:
                self._view_results_button.config(state='disabled')

        all_valid = all(field.is_valid() for field in self._to_protocol_items)
        if all_valid:
            self._generate_prtcl_button.config(state='normal')
        else:
            self._generate_prtcl_button.config(state='disabled')

    @TabContainer.message_decorator('Generating a protocol file', 'Finished generating a protocol file')
    def _generate_protocol(self):
        self._generate_prtcl_button.config(state='disabled')
        print('please wait...')

        bvec_fname = self._bvec_chooser.get_value()
        bval_fname = self._bval_chooser.get_value()
        output_fname = self._output_protocol_chooser.get_value()
        bval_scale = float(self._bval_scale_box.get_value())

        protocol = load_protocol_bval_bvec(bvec=bvec_fname, bval=bval_fname, bval_scale=bval_scale)


        # estimate_st = self._estimate_timings.get_value()
        # if estimate_st:
        #     maxG = float(self._maxG_box.get_value()) / 1000.0
        #
        #     if self._te_box.get_value():
        #         protocol.add_column('TE', float(self._te_box.get_value()) / 1000.0)
        #     if self._Delta_box.get_value():
        #         protocol.add_column('Delta', float(self._Delta_box.get_value()) / 1000.0)
        #     if self._delta_box.get_value():
        #         protocol.add_column('delta', float(self._delta_box.get_value()) / 1000.0)
        #
        #     protocol.add_estimated_protocol_params(maxG=maxG)

        mdt.protocols.write_protocol(protocol, output_fname)

        TabContainer.last_used_protocol = output_fname

        self._view_results_button.config(state='normal')
        self._generate_prtcl_button.config(state='normal')

    def _view_results(self):
        output_fname = self._output_protocol_chooser.get_value()
        t = Tkinter.Toplevel(self.window)
        w = Tkinter.Text(t)
        if os.path.isfile(output_fname):
            w.insert(INSERT, self._format_columns(open(output_fname, 'r').read()))
        w.pack(side='top', fill='both', expand=True)
        t.wm_title('Generated protocol (read-only)')

    def _format_columns(self, table):
        return table.replace("\t", "\t" * 4)


class ProtocolExtraOptionsWindow(SubWindow):

    def __init__(self, parent):
        super(ProtocolExtraOptionsWindow, self).__init__('Extra protocol options')
        self._parent = parent

    def render(self, window):
        subframe = ttk.Frame(window)
        subframe.config(padding=(10, 13, 10, 10))
        subframe.grid_columnconfigure(3, weight=1)

        self._seq_timing_fields = self._get_sequence_timing_fields(subframe)

        button_frame = self._get_button_frame(subframe, window)

        next_row = IntegerGenerator()
        label = ttk.Label(subframe, text="Extra protocol options", font=(None, 14))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        label = ttk.Label(subframe, text="Add extra columns to the generated protocol file.",
                          font=(None, 9, 'italic'))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        self._get_sequence_timing_switch(subframe).render(next_row())
        for field in self._seq_timing_fields:
            field.render(next_row())

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        button_frame.grid(row=next_row(), sticky='W', pady=(10, 0), columnspan=4)

        subframe.pack(fill=BOTH, expand=YES)

    def _get_sequence_timing_switch(self, frame):
        self._estimate_timings = YesNonWidget(
            frame,
            'estimate_sequence_timing',
            self._onchange_cb,
            'Add sequence timings: ',
            '(By default it will guess the sequence timings\n (G, Delta, delta) from the b values)')
        return self._estimate_timings

    def _get_sequence_timing_fields(self, frame):
        self._maxG_box = TextboxWidget(
            frame,
            'maxG_box',
            self._onchange_cb,
            'Max G: ', '(Specify the maximum gradient\n amplitude (mT/m))',
            default_val='40', state='disabled')

        self._Delta_box = TextboxWidget(
            frame,
            'Delta_box',
            self._onchange_cb,
            'Big Delta: ', '(Optionally, use this Delta\n for the sequence timings (ms))',
            default_val='', state='disabled')

        self._delta_box = TextboxWidget(
            frame,
            'delta_box',
            self._onchange_cb,
            'Small delta: ', '(Optionally, use this delta\n for the sequence timings (ms))',
            default_val='', state='disabled')

        self._te_box = TextboxWidget(
            frame,
            'te_box',
            self._onchange_cb,
            'TE: ', '(Optionally, use this TE\n for the echo time (ms))',
            default_val='', state='disabled')

        return [self._maxG_box, self._Delta_box, self._delta_box, self._te_box]

    def _get_button_frame(self, parent, window):
        def accept():
            window.destroy()

        button_frame = ttk.Frame(parent)
        ok_button = ttk.Button(button_frame, text='Accept', command=accept, state='normal')
        cancel_button = ttk.Button(button_frame, text='Cancel', command=window.destroy, state='normal')
        ok_button.grid(row=0)
        cancel_button.grid(row=0, column=1, padx=(10, 0))

        return button_frame

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key

        if id_key == 'estimate_sequence_timing':
            we_will_estimate_seq_timings = calling_widget.get_value()
            if we_will_estimate_seq_timings:
                for field in self._seq_timing_fields:
                    field.set_state('normal')
            else:
                for field in self._seq_timing_fields:
                    field.set_state('disabled')


class ConcatenateShellsTab(TabContainer):

    def __init__(self, window):
        super(ConcatenateShellsTab, self).__init__(window, 'Concatenate shells')

        self._image_1_chooser = FileBrowserWidget(
            self._tab,
            'image_1_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select 4d image 1: ', '(Select the first measured 4d image)')

        self._protocol_1_chooser = FileBrowserWidget(
            self._tab,
            'protocol_1_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select protocol 1: ', '(Select the first protocol file)')

        self._image_2_chooser = FileBrowserWidget(
            self._tab,
            'image_2_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select 4d image 2: ', '(Select the second measured 4d image)')

        self._protocol_2_chooser = FileBrowserWidget(
            self._tab,
            'protocol_2_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select protocol 2: ', '(Select the second protocol file)')

        self._output_image_chooser = FileBrowserWidget(
            self._tab,
            'output_image_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('image_volumes'),
            'Select output volume: ', '(Default is <vol_1><vol_2>.nii.gz)', dialog_type=FileBrowserWidget.SAVE)

        self._output_protocol_chooser = FileBrowserWidget(
            self._tab,
            'output_protocol_chooser',
            self._onchange_cb,
            FileBrowserWidget.common_file_types('protocol_files'),
            'Select output protocol: ', '(Default is <prtcl_1>_<prtcl_2>.prtcl)', dialog_type=FileBrowserWidget.SAVE)

        self._validate_fields = [self._image_1_chooser, self._protocol_1_chooser,
                                 self._image_2_chooser, self._protocol_2_chooser,
                                 self._output_image_chooser, self._output_protocol_chooser]

        self._concatenate_button = ttk.Button(self._tab, text='Concatenate', command=self._apply_concatenate,
                                              state='disabled')

    def get_tab(self):
        next_row = IntegerGenerator()
        
        label = ttk.Label(self._tab, text="Concatenate shells", font=(None, 14))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)
        
        label = ttk.Label(self._tab, text="Concatenate multiple aligned 4d image series of the same subject. ",
                          font=(None, 9, 'italic'))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)
        
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        self._image_1_chooser.render(next_row())
        self._protocol_1_chooser.render(next_row())
        
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(8, 3))
        self._image_2_chooser.render(next_row())
        self._protocol_2_chooser.render(next_row())
        
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(8, 3))
        self._output_image_chooser.render(next_row())
        self._output_protocol_chooser.render(next_row())
        
        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(10, 3))
        self._concatenate_button.grid(row=next_row(), sticky='W', pady=(10, 0))

        return self._tab

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key

        for ind in (1, 2):
            if id_key == 'image_' + str(ind) + '_chooser':
                if calling_widget.is_valid():
                    folder, img_name, ext = split_image_path(calling_widget.get_value())

                    possible_prtcl_path = os.path.join(folder, img_name + '.prtcl')
                    if os.path.exists(possible_prtcl_path):
                        prtcl_chooser = self.__getattribute__('_protocol_' + str(ind) + '_chooser')
                        prtcl_chooser.initial_file = possible_prtcl_path

        if (id_key == 'image_1_chooser' or id_key == 'image_2_chooser') \
                and self._image_1_chooser.is_valid() and self._image_2_chooser.is_valid():

            path, img1_name = split_image_path(self._image_1_chooser.get_value())[0:2]
            img2_name = split_image_path(self._image_2_chooser.get_value())[1]

            self._output_image_chooser.initial_file = os.path.join(path, img1_name + '_' + img2_name + '.nii.gz')

        if (id_key == 'protocol_1_chooser' or id_key == 'protocol_2_chooser')\
                and self._protocol_1_chooser.is_valid() and self._protocol_2_chooser.is_valid():

            prtcl_1_name = os.path.splitext(self._protocol_1_chooser.get_value())[0]
            prtcl_2_name = os.path.basename(os.path.splitext(self._protocol_2_chooser.get_value())[0])

            self._output_protocol_chooser.initial_file = prtcl_1_name + '_' + prtcl_2_name + '.prtcl'

        all_valid = all(field.is_valid() for field in self._validate_fields)
        if all_valid:
            self._concatenate_button.config(state='normal')
        else:
            self._concatenate_button.config(state='disabled')

    def _apply_concatenate(self):
        @TabContainer.message_decorator('Started concatenating shells', 'Finished concatenating shells')
        def concatenate_shells():
            self._concatenate_button.config(state='disabled')

            item1 = {'volume': self._image_1_chooser.get_value(),
                     'protocol': self._protocol_1_chooser.get_value()}

            item2 = {'volume': self._image_2_chooser.get_value(),
                     'protocol': self._protocol_2_chooser.get_value()}

            output_protocol = self._output_protocol_chooser.get_value()
            output_img = self._output_image_chooser.get_value()

            items = (item1, item2)
            concatenate_mri_sets(items, output_img, output_protocol)

            TabContainer.last_used_dwi_image = output_img
            TabContainer.last_used_protocol = output_protocol

            self._concatenate_button.config(state='normal')

        concatenate_thread = threading.Thread(target=concatenate_shells)
        concatenate_thread.start()


class ViewResultsTab(TabContainer):

    def __init__(self, window):
        super(ViewResultsTab, self).__init__(window, 'View results')

        self._parameter_files = {}

        self._input_dir = DirectoryBrowserWidget(
            self._tab, 'input_dir_chooser', self._onchange_cb, True,
            'Select input folder: ', '(Choose a directory containing .nii files)')

        self._maps_chooser = ListboxWidget(
            self._tab, 'maps_chooser', self._onchange_cb, (), '', EXTENDED, 10, 'Select maps: ',
            '(Select the maps you would like to display)')

        self._validate_fields = [self._input_dir]
        self._view_slices_button = ttk.Button(self._tab, text='View', command=self._view_slices,
                                              state='disabled')

    def get_tab(self):
        next_row = IntegerGenerator()

        label = ttk.Label(self._tab, text="View results", font=(None, 14))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="View all maps in a given folder.",
                          font=(None, 9, 'italic'))
        label.grid(row=next_row(), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(5, 3))
        self._input_dir.render(next_row())

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(8, 3))
        self._maps_chooser.render(next_row())

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next_row(), columnspan=5, sticky="EW", pady=(10, 3))
        self._view_slices_button.grid(row=next_row(), sticky='W', pady=(10, 0))

        return self._tab

    def tab_selected(self):
        base_folder = TabContainer.last_used_model_output_folder
        if base_folder and os.path.isdir(base_folder):

            last_model = ''
            if TabContainer.last_optimized_model:
                last_model = TabContainer.last_optimized_model.replace('(Cascade)', '').strip()

            complete_path = os.path.join(base_folder, last_model)
            if os.path.isdir(complete_path):
                self._input_dir.initial_dir = complete_path
            else:
                self._input_dir.initial_dir = base_folder

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        if calling_widget.id_key == 'input_dir_chooser':
            folder = calling_widget.get_value()
            result_files = glob.glob(os.path.join(folder, '*.nii*'))

            def get_name(img_path):
                return split_image_path(os.path.basename(img_path))[1]
            self._parameter_files = {get_name(f): get_name(f) for f in result_files}

            items_list = sorted(self._parameter_files.keys())
            selected_items = filter(lambda v: all(m not in v for m in ('eig', '.d', '.sigma')), items_list)

            self._maps_chooser.set_items(items_list, default_items=selected_items)

        all_valid = all(field.is_valid() for field in self._validate_fields)
        if all_valid:
            self._view_slices_button.config(state='normal')
        else:
            self._view_slices_button.config(state='disabled')

    def _view_slices(self):
        def results_slice_cb():
            selected_maps = self._maps_chooser.get_value()

            if not selected_maps:
                params_list = self._parameter_files.values()
            else:
                params_list = [self._parameter_files[ind] for ind in selected_maps]

            view_results_slice(self._input_dir.get_value(), maps_to_show=params_list)

        view_process = multiprocessing.Process(target=results_slice_cb, args=())
        view_process.start()


class CompositeWidget(object):

    def __init__(self, root_window, id_key, onchange_cb):
        super(CompositeWidget, self).__init__()
        self.id_key = id_key
        self._root_window = root_window
        self._onchange_cb = self._get_cb_function(onchange_cb)

    def _get_cb_function(self, onchange_cb):
        """Generate the final callback function.

        This wraps the given callback function in a new function that adds a reference to the calling composite widget
        (this) as first argument to the callback.

        Args:
            onchange_cb: the user provided callback function

        Returns:
            a new callback function that adds this calling widget as first argument to the function
        """
        def extra_cb(*args, **kwargs):
            onchange_cb(self, *args, **kwargs)
        return extra_cb

    def is_valid(self):
        pass

    def get_value(self):
        pass

    def render(self, row):
        pass


class FileBrowserWidget(CompositeWidget):

    OPEN = 1
    SAVE = 2

    def __init__(self, root_window, id_key, onchange_cb, file_types, label_text, helper_text, dialog_type=None):
        super(FileBrowserWidget, self).__init__(root_window, id_key, onchange_cb)

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

    def __init__(self, root_window, id_key, onchange_cb, must_exist, label_text, helper_text):
        super(DirectoryBrowserWidget, self).__init__(root_window, id_key, onchange_cb)

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

    def __init__(self, root_window, id_key, onchange_cb, label_text, helper_text, default_val='',
                 required=False, state='normal'):
        super(TextboxWidget, self).__init__(root_window, id_key, onchange_cb)

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

    def __init__(self, root_window, id_key, onchange_cb, items, default_item, label_text, helper_text):
        super(DropdownWidget, self).__init__(root_window, id_key, onchange_cb)

        self._label_text = label_text
        self._helper_text = helper_text

        self._chooser_var = StringVar(self._root_window)
        self._chooser = ttk.OptionMenu(self._root_window, self._chooser_var, default_item, *items,
                                       command=self._onchange_cb)

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

    def __init__(self, root_window, id_key, onchange_cb, items, default_item, selectmode, height, label_text, helper_text):
        super(ListboxWidget, self).__init__(root_window, id_key, onchange_cb)

        self._selectbox_height = height
        self._label_text = label_text
        self._helper_text = helper_text

        self._chooser_frame = ttk.Frame(self._root_window)

        self._scrollbar = ttk.Scrollbar(self._chooser_frame, orient=VERTICAL)
        self._chooser = Listbox(self._chooser_frame, selectmode=selectmode, exportselection=0,
                                yscrollcommand=self._mouse_scroll_detect, height=height)
        self._chooser.bind('<<ListboxSelect>>', self._onchange_cb)
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

    def __init__(self, root_window, id_key, onchange_cb, label_text, helper_text, default_val=0):
        super(YesNonWidget, self).__init__(root_window, id_key, onchange_cb)

        self._label_text = label_text
        self._helper_text = helper_text

        self._chooser_var = IntVar(self._root_window)
        self._yes = ttk.Radiobutton(self._root_window, text='Yes', variable=self._chooser_var,
                                    value=1, command=self._onchange_cb)
        self._no = ttk.Radiobutton(self._root_window, text='No', variable=self._chooser_var,
                                   value=0, command=self._onchange_cb)
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

    def __init__(self, root_window, id_key, subwindow, label_text, helper_text):
        super(SubWindowWidget, self).__init__(root_window, id_key, None)
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
        t.resizable(width=FALSE, height=FALSE)


class ScrolledText(Tkinter.Text):
    """Copied from ScrolledText from TK"""
    def __init__(self, master=None, **kw):
        self.frame = Tkinter.Frame(master)

        yscroll = Tkinter.Scrollbar(self.frame, orient=VERTICAL)
        xscroll = Tkinter.Scrollbar(self.frame, orient=HORIZONTAL)

        kw.update({'yscrollcommand': yscroll.set})
        kw.update({'xscrollcommand': xscroll.set})
        Tkinter.Text.__init__(self, self.frame, **kw)

        yscroll['command'] = self.yview
        xscroll['command'] = self.xview

        yscroll.grid(column=1, row=0, sticky="news")
        xscroll.grid(column=0, row=1, sticky="ew")
        self.grid(column=0, row=0, sticky="news")

        self.frame.grid()
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        text_meths = vars(Tkinter.Text).keys()
        methods = vars(Tkinter.Pack).keys() + vars(Tkinter.Grid).keys() + vars(Tkinter.Place).keys()
        methods = set(methods).difference(text_meths)

        for m in methods:
            if m[0] != '_' and m != 'config' and m != 'configure':
                setattr(self, m, getattr(self.frame, m))

    def __str__(self):
        return str(self.frame)