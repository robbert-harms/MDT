from collections import OrderedDict
import six

try:
    #python 2.7
    from Tkconstants import W, HORIZONTAL, BOTH, YES, INSERT
    import ttk
    from Tkinter import Toplevel
except ImportError:
    # python 3.4
    from tkinter.constants import W, HORIZONTAL, BOTH, YES, INSERT
    from tkinter import ttk
    from tkinter import Toplevel

import copy
from itertools import count
from math import log10
import os
from numpy import genfromtxt
import numpy as np
from mdt import load_protocol_bval_bvec
import mdt
from mdt.gui.tk.utils import TabContainer, SubWindow
from mdt.gui.tk.widgets import FileBrowserWidget, TextboxWidget, SubWindowWidget, ScrolledText, YesNonWidget, \
    RadioButtonWidget
from mdt.gui.utils import ProtocolOptions, function_message_decorator

__author__ = 'Robbert Harms'
__date__ = "2015-08-26"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateProtocolFileTab(TabContainer):

    def __init__(self, window, cl_process_queue, output_queue):
        super(GenerateProtocolFileTab, self).__init__(window, cl_process_queue, output_queue, 'Generate protocol file')

        self.protocol_options = ProtocolOptions()

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
            'B-value scale factor: ', '(We expect the b-values in the\noutput protocol in units of s/m^2)',
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
        row_nmr = count()
        label = ttk.Label(self._tab, text="Generate protocol file", font=(None, 14))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        label = ttk.Label(self._tab, text="Create a protocol file containing all your sequence information.",
                          font=(None, 9, 'italic'))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        for field in self._to_protocol_items:
            field.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(8, 3))
        self._extra_options_button.render(next(row_nmr))

        ttk.Separator(self._tab, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(10, 3))
        self._buttons_frame.grid(row=next(row_nmr), sticky='W', columnspan=4, pady=(10, 0))
        return self._tab

    def _onchange_cb(self, calling_widget, *args, **kwargs):
        id_key = calling_widget.id_key

        self._update_global_initial_dir(calling_widget, ['bvec_chooser', 'bval_chooser'])

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

    @function_message_decorator('Generating a protocol file', 'Finished generating a protocol file')
    def _generate_protocol(self):
        self._generate_prtcl_button.config(state='disabled')
        print('generating protocol...')

        bvec_fname = self._bvec_chooser.get_value()
        bval_fname = self._bval_chooser.get_value()
        output_fname = self._output_protocol_chooser.get_value()
        bval_scale = float(self._bval_scale_box.get_value())

        protocol = load_protocol_bval_bvec(bvec=bvec_fname, bval=bval_fname, bval_scale=bval_scale)

        if self.protocol_options.estimate_sequence_timings:
            protocol.add_estimated_protocol_params(maxG=self.protocol_options.maxG)
            for column in self.protocol_options.extra_column_names:
                value = self.protocol_options.__getattribute__(column)
                self._add_column_to_protocol(protocol, column, value, self.protocol_options.seq_timings_units)

        mdt.protocols.write_protocol(protocol, output_fname)

        TabContainer.last_used_protocol = output_fname

        self._view_results_button.config(state='normal')
        self._generate_prtcl_button.config(state='normal')

    def _view_results(self):
        output_fname = self._output_protocol_chooser.get_value()
        toplevel = Toplevel(self.window)

        txt_frame = ttk.Frame(toplevel)
        w = ScrolledText(txt_frame, wrap='none')
        w.pack(fill=BOTH, expand=YES)
        txt_frame.pack(fill=BOTH, expand=YES)

        if os.path.isfile(output_fname):
            w.insert(INSERT, self._format_columns(open(output_fname, 'r').read()))
        toplevel.wm_title('Generated protocol (read-only)')

        width = 800
        height = 500
        x = (toplevel.winfo_screenwidth() // 2) - (width // 2)
        y = (toplevel.winfo_screenheight() // 2) - (height // 2)
        toplevel.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def _format_columns(self, table):
        return table.replace("\t", "\t" * 4)

    def _add_column_to_protocol(self, protocol, column, value, units):
        """Adds the given value to the protocol as the given column name.

        Args:
            protocol (Protocol): the protocol object
            column (str): the name of the column
            value (float, str): either a single column value, or a string which we will try to interpret as a file.
            units (str): either 's', or 'ms'
        """
        mult_factor = 1e-3 if units == 'ms' else 1

        if value is not None:
            if os.path.isfile(value):
                protocol.add_column_from_file(column, value, mult_factor)
            else:
                protocol.add_column(column, float(value) * mult_factor)


class ProtocolExtraOptionsWindow(SubWindow):

    def __init__(self, parent):
        super(ProtocolExtraOptionsWindow, self).__init__('Extra protocol options')
        self._parent = parent
        self._protocol_options = copy.copy(self._parent.protocol_options)

    def render(self, window):
        subframe = ttk.Frame(window)
        subframe.config(padding=(10, 13, 10, 10))
        subframe.grid_columnconfigure(3, weight=1)

        self._seq_timing_fields = self._get_sequence_timing_fields(subframe)

        button_frame = self._get_button_frame(subframe, window)

        row_nmr = count()
        label = ttk.Label(subframe, text="Extra protocol options", font=(None, 14))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        label = ttk.Label(subframe, text="Add extra columns to the generated protocol file.",
                          font=(None, 9, 'italic'))
        label.grid(row=next(row_nmr), column=0, columnspan=4, sticky=W)

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        self._get_sequence_timing_switch(subframe).render(next(row_nmr))
        for field in self._seq_timing_fields:
            field.set_state('normal' if self._estimate_timings.get_value() else 'disabled')
            field.render(next(row_nmr))

        ttk.Separator(subframe, orient=HORIZONTAL).grid(row=next(row_nmr), columnspan=5, sticky="EW", pady=(5, 3))
        button_frame.grid(row=next(row_nmr), sticky='W', pady=(10, 0), columnspan=4)

        subframe.pack(fill=BOTH, expand=YES)

    def _get_sequence_timing_switch(self, frame):
        self._estimate_timings = YesNonWidget(
            frame,
            'estimate_sequence_timing',
            self._onchange_cb,
            'Add sequence timings: ',
            '(By default it will guess the sequence timings\n (G, Delta, delta) from the b values)',
            default_val=self._protocol_options.estimate_sequence_timings)
        return self._estimate_timings

    def _get_sequence_timing_fields(self, frame):
        state = 'normal' if self._protocol_options.estimate_sequence_timings else 'disabled'

        self._units_box = RadioButtonWidget(
            frame,
            'seq_timings_units',
            self._onchange_cb,
            'Sequence timing units: ',
            '(The units in which we specify Delta, delta and TE)',
            OrderedDict([('Seconds (s)', 's'), ('Milliseconds (ms)', 'ms')]),
            default_val=self._protocol_options.seq_timings_units)

        self._maxG_box = TextboxWidget(
            frame,
            'maxG',
            self._onchange_cb,
            'Max G: ', '(Specify the maximum gradient\n amplitude (T/m))',
            default_val=self._protocol_options.maxG, state=state)

        self._Delta_box = TextboxWidget(
            frame,
            'Delta',
            self._onchange_cb,
            'Big Delta: ', '(Optional Delta, give a filename or number)',
            default_val=self._protocol_options.Delta, state=state)

        self._delta_box = TextboxWidget(
            frame,
            'delta',
            self._onchange_cb,
            'Small delta: ', '(Optional delta, give a filename or number)',
            default_val=self._protocol_options.delta, state=state)

        self._te_box = TextboxWidget(
            frame,
            'TE',
            self._onchange_cb,
            'TE: ', '(Optional TE, give a filename or number)',
            default_val=self._protocol_options.TE, state=state)

        return [self._units_box, self._maxG_box, self._Delta_box, self._delta_box, self._te_box]

    def _get_button_frame(self, parent, window):
        def accept():
            self._parent.protocol_options = self._protocol_options
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

            self._protocol_options.estimate_sequence_timings = we_will_estimate_seq_timings

        else:
            for field in self._seq_timing_fields:
                if field.id_key == id_key:
                    setattr(self._protocol_options, field.id_key, calling_widget.get_value())
