from Tkconstants import W, HORIZONTAL
from itertools import count
import os
import threading
import ttk
from mdt import concatenate_mri_sets
from mdt.gui.tk.utils import TabContainer
from mdt.gui.tk.widgets import FileBrowserWidget
from mdt.gui.utils import function_message_decorator
from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2015-08-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ConcatenateShellsTab(TabContainer):

    def __init__(self, window, cl_process_queue):
        super(ConcatenateShellsTab, self).__init__(window, cl_process_queue, 'Concatenate shells')

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
        row_nmr = count()

        label = ttk.Label(self._tab, text="Concatenate shells", font=(None, 14))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="Concatenate multiple aligned 4d image series of the same subject. ",
                          font=(None, 9, 'italic'))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        self._image_1_chooser.render(next(row_nmr))
        self._protocol_1_chooser.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(8, 3))
        self._image_2_chooser.render(next(row_nmr))
        self._protocol_2_chooser.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(8, 3))
        self._output_image_chooser.render(next(row_nmr))
        self._output_protocol_chooser.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(10, 3))
        self._concatenate_button.grid(row=next(row_nmr), sticky='W', pady=(10, 0))

        return self._tab

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key

        self._update_global_initial_dir(calling_widget, ['image_1_chooser', 'protocol_1_chooser', 'image_2_chooser',
                                                         'protocol_2_chooser', 'output_image_chooser',
                                                         'output_protocol_chooser'])

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
        @function_message_decorator('Started concatenating shells', 'Finished concatenating shells')
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
        concatenate_shells()