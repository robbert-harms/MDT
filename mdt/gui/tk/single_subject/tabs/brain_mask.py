from Queue import Empty
from Tkconstants import W, HORIZONTAL
from itertools import count
import os
import ttk
import multiprocessing
import numpy as np
from mdt import create_median_otsu_brain_mask, load_dwi, load_brain_mask
from mdt.gui.tk.widgets import FileBrowserWidget, TextboxWidget
from mdt.gui.tk.utils import TabContainer
from mdt.gui.utils import function_message_decorator
from mdt.utils import split_image_path
from mdt.visualization import MapsVisualizer

__author__ = 'Robbert Harms'
__date__ = "2015-08-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateBrainMaskTab(TabContainer):

    def __init__(self, window, cl_process_queue):
        super(GenerateBrainMaskTab, self).__init__(window, cl_process_queue, 'Generate brain mask')

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
        row_nmr = count()

        label = ttk.Label(self._tab, text="Generate brain mask", font=(None, 14))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="Creates a mask for a whole brain using the median-otsu algorithm.",
                          font=(None, 9, 'italic'))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))

        for field in self._mask_items:
            field.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))

        self._whole_mask_buttons_frame.grid(row=next(row_nmr), sticky='W', pady=(10, 0), columnspan=4)
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

            possible_prtcl = os.path.join(path, img_name + '.prtcl')
            if os.path.isfile(possible_prtcl):
                self._protocol_file_chooser.initial_file = possible_prtcl

        self._update_global_initial_dir(calling_widget, ['image_vol_chooser', 'protocol_file_chooser',
                                                         'output_bm_chooser'])

        if os.path.isfile(self._output_bm_chooser.get_value()) and self._image_vol_chooser.is_valid():
            self._view_brain_mask_button.config(state='normal')
        else:
            self._view_brain_mask_button.config(state='disabled')

        if self._image_vol_chooser.is_valid() and self._output_bm_chooser.is_valid() and \
                self._protocol_file_chooser.is_valid():
            self._run_whole_brain_mask_button.config(state='normal')
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

        manager = multiprocessing.Manager()
        finish_queue = manager.Queue()
        proc = CreateMaskProc(finish_queue, volume, prtcl, output_fname, filter_mean_radius,
                              filter_passes, threshold)

        self._cl_process_queue.put(proc)

        def _wait_for_mask_completion():
            try:
                finish_queue.get(block=False)
                self._run_whole_brain_mask_button.config(state='normal')
                self._view_brain_mask_button.config(state='normal')
            except Empty:
                self.window.after(100, _wait_for_mask_completion)

        self.window.after(100, _wait_for_mask_completion)

        TabContainer.last_used_dwi_image = volume
        TabContainer.last_used_protocol = prtcl
        TabContainer.last_used_mask = output_fname

    def _view_brain_mask(self):
        mask = self._output_bm_chooser.get_value()
        dwi_path = self._image_vol_chooser.get_value()

        if os.path.isfile(mask) and os.path.isfile(dwi_path):
            proc = ViewMaskProcess(dwi_path, mask)
            proc.start()


class CreateMaskProc(object):

    def __init__(self, finish_queue, volume, prtcl, output_fname, filter_mean_radius,
                 filter_passes, threshold):
        self._finish_queue = finish_queue
        self._volume = volume
        self._prtcl = prtcl
        self._output_fname = output_fname
        self._filter_mean_radius = filter_mean_radius
        self._filter_passes = filter_passes
        self._threshold = threshold

    @function_message_decorator('Started creating a mask.', 'Finished creating a mask.')
    def __call__(self, *args, **kwargs):
        create_median_otsu_brain_mask(self._volume, self._prtcl, self._output_fname,
                                      median_radius=self._filter_mean_radius, numpass=self._filter_passes,
                                      mask_threshold=self._threshold)
        self._finish_queue.put('DONE')


class ViewMaskProcess(multiprocessing.Process):

    def __init__(self, dwi_path, brain_mask_path):
        super(ViewMaskProcess, self).__init__()
        self._dwi_path = dwi_path
        self._brain_mask_path = brain_mask_path

    def run(self):
        image_data = load_dwi(self._dwi_path)[0]
        mask = np.expand_dims(load_brain_mask(self._brain_mask_path), axis=3)
        masked_image = image_data * mask
        s = image_data.shape

        MapsVisualizer({'Masked': masked_image,
                        'DWI': image_data}).show(dimension=1, slice_ind=s[1]/2)