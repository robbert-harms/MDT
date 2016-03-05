try:
    #python 2.7
    from Queue import Queue
    from Queue import Empty
except ImportError:
    # python 3.4
    from queue import Queue
    from queue import Empty

import time
import multiprocessing

__author__ = 'Robbert Harms'
__date__ = "2015-08-27"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class StdRedirect(object):

    def __init__(self, queue):
        self._queue = queue

    def write(self, message):
        self._queue.put(message)

    def flush(self):
        pass


def cl_process_runner(input_queue, output_queue):
    import pyopencl
    import mdt
    import mot
    import sys
    from mdt.log_handlers import LogDispatchHandler
    from mdt.gui.utils import ForwardingListener

    LogDispatchHandler.add_listener(ForwardingListener(output_queue))

    redirect = StdRedirect(output_queue)
    sys.stdout = redirect
    sys.stderr = redirect

    while True:
        try:
            func = input_queue.get(False)

            if func is None:
                break

            try:
                func()
            except:
                import traceback
                import sys
                sys.stderr.write('Process runner crashed')
                sys.stderr.flush()
                traceback.print_exc()
        except Empty:
            pass

        time.sleep(0.01)


def start_single_gui(*args, **kwargs):
    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()

    mp = multiprocessing.Pool(1, cl_process_runner, (input_queue, output_queue))

    from mdt.gui import tkgui
    window = tkgui.get_window(input_queue, output_queue, *args, **kwargs)
    window.mainloop()

    input_queue.put(None)

    mp.close()
    mp.join()
