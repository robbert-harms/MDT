try:
    #python 2.7
    from Tkconstants import BOTH, FALSE, YES
    from Tkinter import Tk, TclError, PhotoImage
    import ttk
except ImportError:
    # python 3.4
    from tkinter.constants import BOTH, FALSE, YES
    from tkinter import Tk, TclError, PhotoImage
    from tkinter import ttk

import sys
from pkg_resources import resource_filename
import mdt
from mdt.gui.tk.single_subject.tabs.brain_mask import GenerateBrainMaskTab
from mdt.gui.tk.single_subject.tabs.concatenate_shells import ConcatenateShellsTab
from mdt.gui.tk.single_subject.tabs.protocol import GenerateProtocolFileTab
from mdt.gui.tk.single_subject.tabs.roi_mask import GenerateROIMaskTab
from mdt.gui.tk.single_subject.tabs.run_model import RunModelTab
from mdt.gui.tk.single_subject.tabs.view_results import ViewResultsTab
from mdt.gui.tk.widgets import CompositeWidget, LoggingTextArea
from mdt.gui.utils import print_welcome_message, LogMonitorThread
import mdt.utils
import mdt.protocols
import mdt.configuration

__author__ = 'Robbert Harms'
__date__ = "2014-10-01"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


mdt.utils.setup_logging(disable_existing_loggers=True)


def get_window(input_queue, output_queue, initial_dir=None):
    return ToolkitGUIWindow(input_queue, output_queue, initial_dir=initial_dir)


class ToolkitGUIWindow(Tk):

    def __init__(self, input_queue, output_queue, initial_dir=None):
        Tk.__init__(self)

        self._cl_process_queue = input_queue
        self._output_queue = output_queue

        self._stdout_old = sys.stdout
        self._stderr_old = sys.stderr

        self._set_style()
        self._set_icon()
        self._set_size_and_position()

        self.wm_title("Maastricht Diffusion Toolbox")
        self.resizable(width=FALSE, height=FALSE)
        self.update_idletasks()

        txt_frame = ttk.Frame(self)
        self._log_box = LoggingTextArea(txt_frame)
        self._log_box.pack(fill=BOTH, expand=YES)

        notebook = MainNotebook(self, self._cl_process_queue, self._output_queue)
        notebook.pack(fill=BOTH, expand=YES)

        txt_frame.pack(fill=BOTH, expand=YES)

        self.after(100, self._window_start_cb)

        self.protocol("WM_DELETE_WINDOW", self._window_close_cb)
        CompositeWidget.initial_dir = initial_dir

    def _window_start_cb(self):
        sys.stdout = self._log_box
        sys.stderr = self._log_box

        print_welcome_message()

        self._monitor = LogMonitorThread(self._output_queue, self._log_box)
        self._monitor.daemon = True
        self._monitor.start()

    def _window_close_cb(self):
        sys.stdout = self._stdout_old
        sys.stderr = self._stderr_old

        self._monitor.send_stop_signal()
        self._monitor.join()

        self.destroy()

    def _set_style(self):
        s = ttk.Style()
        try:
            s.theme_use('clam')
        except TclError:
            pass

    def _set_icon(self):
        img = PhotoImage(file=resource_filename('mdt', 'data/logo.gif'))
        self.tk.call('wm', 'iconphoto', self._w, img)

    def _set_size_and_position(self):
        width = 900
        height = 630
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))


class MainNotebook(ttk.Notebook):

    def __init__(self, window, cl_process_queue, output_queue):
        ttk.Notebook.__init__(self, window)

        self.tabs = [RunModelTab(window, cl_process_queue, output_queue),
                     GenerateBrainMaskTab(window, cl_process_queue, output_queue),
                     GenerateROIMaskTab(window, cl_process_queue, output_queue),
                     GenerateProtocolFileTab(window, cl_process_queue, output_queue),
                     ConcatenateShellsTab(window, cl_process_queue, output_queue),
                     ViewResultsTab(window, cl_process_queue, output_queue)]

        for tab in self.tabs:
            self.add(tab.get_tab(), text=tab.tab_name)

        self.bind_all("<<NotebookTabChanged>>", self._tab_change_cb)

    def _tab_change_cb(self, event):
        self.tabs[event.widget.index("current")].tab_selected()