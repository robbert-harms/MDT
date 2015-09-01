try:
    #python 2.7
    from Tkconstants import BROWSE, W, HORIZONTAL
    import ttk
    import tkMessageBox
except ImportError:
    # python 3.4
    from tkinter.constants import BROWSE, W, HORIZONTAL
    from tkinter import ttk
    import tkinter.messagebox as tkMessageBox


from itertools import count
import os
import nibabel as nib
from nibabel.spatialimages import ImageFileError
from mdt import create_slice_roi
from mdt.gui.tk.utils import TabContainer
from mdt.gui.tk.widgets import FileBrowserWidget, DropdownWidget, ListboxWidget
from mdt.gui.utils import function_message_decorator

__author__ = 'Robbert Harms'
__date__ = "2015-08-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateROIMaskTab(TabContainer):

    def __init__(self, window, cl_process_queue, output_queue):
        super(GenerateROIMaskTab, self).__init__(window, cl_process_queue, output_queue, 'Generate ROI mask')

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
        row_nmr = count()

        label = ttk.Label(self._tab, text="Generate ROI mask", font=(None, 14))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="Create a brain mask with a Region Of Interest "
                                          "that only includes the voxels in the selected slice. ",
                          font=(None, 9, 'italic'))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))

        for field in self._roi_items:
            field.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(10, 3))

        self._buttons_frame.grid(row=next(row_nmr), sticky='W', pady=(10, 0), columnspan=4)
        return self._tab

    def tab_selected(self):
        self._init_path_chooser(self._output_roi_chooser, TabContainer.last_used_roi_mask)
        self._init_path_chooser(self._brain_mask_vol_chooser, TabContainer.last_used_mask)

        if TabContainer.last_used_image_dimension is not None:
            if TabContainer.last_used_image_dimension in self._dimensions.values():
                self._roi_dimension_index.set_value(TabContainer.last_used_image_dimension)

        if TabContainer.last_used_image_slice_ind is not None and self._dimensions:
            if TabContainer.last_used_image_slice_ind < \
                    self._dimension_shape[self._dimensions[self._roi_dimension_index.get_value()]]:
                self._roi_slice_index.set_default_ind(TabContainer.last_used_image_slice_ind)

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key
        brain_mask_fname = self._brain_mask_vol_chooser.get_value()

        self._update_global_initial_dir(calling_widget, ['brain_mask_chooser'])

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

    @function_message_decorator('Started with generating a slice ROI', 'Finished generating a slice ROI')
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
        TabContainer.last_used_image_dimension = roi_dim
        TabContainer.last_used_image_slice_ind = roi_slice

        self._run_slice_roi_button.config(state='normal')