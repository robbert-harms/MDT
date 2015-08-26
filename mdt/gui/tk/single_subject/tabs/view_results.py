from Tkconstants import EXTENDED, W, HORIZONTAL
import glob
from itertools import count
import multiprocessing
import os
import ttk
from mdt import view_results_slice
from mdt.gui.tk.utils import TabContainer
from mdt.gui.tk.widgets import DirectoryBrowserWidget, ListboxWidget
from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2015-08-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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
        row_nmr = count()

        label = ttk.Label(self._tab, text="View results", font=(None, 14))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="View all maps in a given folder.",
                          font=(None, 9, 'italic'))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        self._input_dir.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(8, 3))
        self._maps_chooser.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(10, 3))
        self._view_slices_button.grid(row=next(row_nmr), sticky='W', pady=(10, 0))

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
        selected_maps = self._maps_chooser.get_value()

        if not selected_maps:
            params_list = self._parameter_files.values()
        else:
            params_list = [self._parameter_files[ind] for ind in selected_maps]

        dim = TabContainer.last_used_image_dimension
        slice_ind = TabContainer.last_used_image_slice_ind

        view_process = multiprocessing.Process(
            target=view_results_slice,
            args=[self._input_dir.get_value()],
            kwargs={'maps_to_show': params_list, 'dimension': dim, 'slice_ind': slice_ind})
        view_process.start()