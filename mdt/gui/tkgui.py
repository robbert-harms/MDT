from Tkconstants import BOTH, FALSE, YES
from Tkinter import Tk, TclError
import Tkinter
import ttk
import sys
from pkg_resources import resource_filename
import mdt
from mdt.gui.tk.single_subject.tabs.brain_mask import GenerateBrainMaskTab
from mdt.gui.tk.single_subject.tabs.concatenate_shells import ConcatenateShellsTab
from mdt.gui.tk.single_subject.tabs.protocol import GenerateProtocolFileTab
from mdt.gui.tk.single_subject.tabs.roi_mask import GenerateROIMaskTab
from mdt.gui.tk.single_subject.tabs.run_model import RunModelTab
from mdt.gui.tk.single_subject.tabs.view_results import ViewResultsTab
from mdt.gui.tk.utils import TextboxLogListener
from mdt.gui.tk.widgets import ScrolledText, CompositeWidget
from mdt.gui.utils import print_welcome_message, update_user_settings
from mdt.log_handlers import LogDispatchHandler
import mdt.utils
import mdt.protocols
import mdt.configuration

__author__ = 'Robbert Harms'
__date__ = "2014-10-01"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


mdt.utils.setup_logging(disable_existing_loggers=True)


def get_window(initial_dir=None):
    return ToolkitGUIWindow(initial_dir=initial_dir)


class ToolkitGUIWindow(Tk):

    def __init__(self, initial_dir=None):
        Tk.__init__(self)

        s = ttk.Style()
        try:
            s.theme_use('clam')
        except TclError:
            pass

        self.wm_title("Diffusion MRI Toolbox")
        self.resizable(width=FALSE, height=FALSE)
        self.update_idletasks()

        img = Tkinter.PhotoImage(file=resource_filename('mdt', 'data/logo.gif'))
        self.tk.call('wm', 'iconphoto', self._w, img)

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
        CompositeWidget.initial_dir = initial_dir

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