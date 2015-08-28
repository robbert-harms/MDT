from Tkconstants import VERTICAL, HORIZONTAL, E, W, END, FALSE, NONE
import Tkinter
from Tkinter import StringVar, Listbox, IntVar
import os
import platform
import tkFileDialog
import ttk

__author__ = 'Robbert Harms'
__date__ = "2015-08-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ScrolledText(Tkinter.Text):

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


class LoggingTextArea(ScrolledText):

    def __init__(self, master=None, **kwargs):
        kwargs.update({'wrap': NONE})
        ScrolledText.__init__(self, master=master, **kwargs)

    def write(self, string):
        self.configure(state='normal')
        self.insert(END, string)
        self.configure(state='disabled')
        self.see(END)

    def flush(self):
        pass


class CompositeWidget(object):

    initial_dir = None

    def __init__(self, root_window, id_key, onchange_cb):
        """Container for all widgets in use."""
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

        if not init_dir:
            init_dir = CompositeWidget.initial_dir

        if init_dir:
            if os.path.isdir(init_dir):
                options['initialdir'] = init_dir
                options['initialfile'] = None
            else:
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
            l = (('All image files', ('*.nii.gz', '*.nii', '*.img')),)
        elif choice == 'protocol_files':
            l = (('Protocol files', '*.prtcl'),)
        elif choice == 'txt':
            l = (('Text files', '*.txt'),)
        elif choice == 'bvec':
            l = (('b-vector files', '*.bvec'),
                 ('Text files', '*.txt'),
                 ('All files', '*.*'))
        elif choice == 'bval':
            l = (('b-values files', '*.bval'),
                 ('Text files', '*.txt'),
                 ('All files', '*.*'))
        else:
            return ('All files', '*.*'),

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
        if not init_dir:
            init_dir = self.initial_dir
        if not init_dir:
            init_dir = CompositeWidget.initial_dir

        if init_dir:
            if os.path.isdir(init_dir):
                options['initialdir'] = init_dir
            else:
                options['initialdir'] = os.path.dirname(init_dir)

        options['parent'] = self._root_window
        options['title'] = self._label_text

        filename = tkFileDialog.askdirectory(**options)
        if filename:
            self._fname_var.set(filename)


class TextboxWidget(CompositeWidget):

    def __init__(self, root_window, id_key, onchange_cb, label_text, helper_text, default_val='',
                 required=False, state='normal'):
        super(TextboxWidget, self).__init__(root_window, id_key, onchange_cb)

        if default_val is None:
            default_val = ''

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

    def set_value(self, value):
        self._chooser_var.set(value)

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

    def set_default_ind(self, default_items):
        try:
            for ind in default_items:
                self._chooser.selection_set(ind)
        except TypeError:
            self._chooser.select_set(default_items)
            self._chooser.see(default_items)

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